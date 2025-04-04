import argparse
from argparse import ArgumentParser
from typing import Tuple, List

from cshogi import Board
from cshogi.dlshogi import FEATURES1_NUM, FEATURES2_NUM, make_move_label
import numpy as np
from numpy.typing import NDArray, DTypeLike
import onnxruntime as ort


def configure_session_args(parser: ArgumentParser) -> ArgumentParser:
    """
    Configure command-line arguments for setting up an inference session.

    Args:
        parser (ArgumentParser): The argument parser to configure.

    Returns:
        ArgumentParser: The configured argument parser.
    """

    parser.add_argument("--model-path", type=str, default="model.onnx", help="ONNX Model path", dest="model_path")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID to be used", dest="device_id")
    parser.add_argument("--enable-cuda", action="store_true", help="Enable CUDAExecutionProvider", dest="enable_cuda")
    parser.add_argument("--enable-tensorrt", action="store_true", help="Enable TensorrtExecutionProvider", dest="enable_tensorrt")

    return parser


def create_session(args: argparse.Namespace) -> ort.InferenceSession:
    """
    Create an ONNX Runtime inference session.

    Args:
        args (argparse.Namespace): Command-line arguments configured by `configure_session_args`.

    Returns:
        ort.InferenceSession: A configured ONNX Runtime inference session.

    Raises:
        RuntimeError: If no valid execution providers are available.
    """

    available_providers = ort.get_available_providers()
    providers = []

    if args.enable_tensorrt and "TensorrtExecutionProvider" in available_providers:
        providers.append(("TensorrtExecutionProvider", {
            "device_id": args.device_id,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
        }))
        print("TensorrtExecutionProvider is enabled.")

    if args.enable_cuda and "CUDAExecutionProvider" in available_providers:
        providers.append(("CUDAExecutionProvider", {
            "device_id": args.device_id,
        }))
        print("CUDAExecutionProvider is enabled.")

    if "CPUExecutionProvider" in available_providers:
        providers.append("CPUExecutionProvider")
        print("CPUExecutionProvider is enabled.")

    if not providers:
        raise RuntimeError("No available providers found.")

    return ort.InferenceSession(args.model_path, providers=providers)


def allocate_input_features(batch_size: int) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Allocate memory for input feature tensors used in batch inference.

    Args:
        batch_size (int): Number of samples in the batch.

    Returns:
        Tuple[NDArray[np.float32], NDArray[np.float32]]:
            - First input feature tensor of shape (batch_size, FEATURES1_NUM, 9, 9).
            - Second input feature tensor of shape (batch_size, FEATURES2_NUM, 9, 9).
    """
    return (np.empty((batch_size, FEATURES1_NUM, 9, 9), dtype=np.float32),
            np.empty((batch_size, FEATURES2_NUM, 9, 9), dtype=np.float32),)


def inference(input_features1: NDArray[np.float32],
              input_features2: NDArray[np.float32],
              session: ort.InferenceSession) -> Tuple[NDArray[np.float32], NDArray[NDArray[np.float32]]]:
    """
    Perform batch inference using ONNX Runtime with IO binding for efficiency.

    Args:
        input_features1 (NDArray[np.float32])  : First input feature tensor of shape (batch_size, FEATURES1_NUM, 9, 9).
        input_features2 (NDArray[np.float32])  : Second input feature tensor of shape (batch_size, FEATURES2_NUM, 9, 9).
        session         (ort.InferenceSession) : ONNX Runtime inference session.

    Returns:
        Tuple[NDArray[np.float32], NDArray[np.float32]]:
            - Output value tensor, reshaped to a 1D array of shape (batch_size,).
            - Output policy logits tensor of shape (batch_size, ...).
    """

    io_binding = session.io_binding()
    io_binding.bind_cpu_input("input1", input_features1)
    io_binding.bind_cpu_input("input2", input_features2)
    io_binding.bind_output("output_policy")
    io_binding.bind_output("output_value")
    session.run_with_iobinding(io_binding)
    outputs = io_binding.copy_outputs_to_cpu()
    values = outputs[1].reshape(-1)
    logits = outputs[0]

    return values, logits


def softmax(logits: NDArray[np.float32], temperature: float):
    """
    Compute the softmax probabilities from raw logits with temperature scaling.

    This function normalizes a vector of logits into a probability distribution.
    Temperature controls the "sharpness" of the distribution:
      - High temperature (>1): makes the distribution more uniform (less confident).
      - Low temperature (<1): makes the distribution peakier (more confident).
      - Temperature of 0 is treated as a very small number (1e-10) to avoid division by zero.

    Args:
        logits     (NDArray[np.float32]): 1D array of raw logits.
        temperature (float)              : Temperature parameter for scaling.

    Returns:
        NDArray[np.float32]: Softmax probabilities corresponding to the logits.
    """

    temperature = temperature if temperature != 0 else 1e-10

    max_logit = np.max(logits)
    exp_logits = np.exp((logits - max_logit) / temperature)
    sum_exp = np.sum(exp_logits)
    probabilities = exp_logits / sum_exp

    return probabilities


def convert_to_score(win_rate: NDArray[np.float32], score_scaling: float) -> NDArray[np.int16]:
    """
    Convert a win rate array to a score representation.

    The function maps win rates to integer scores using the following rules:
      - A win rate of 1.0 maps to a score of 32000.
      - A win rate of 0.0 maps to a score of -32000.
      - Other values are transformed using the formula:
        score = -score_scaling * log(1 / win_rate - 1)

    Args:
        win_rate      (NDArray[np.float32]): Array of win rates in the range [0,1].
        score_scaling (float)             : Scaling factor for score calculation.

    Returns:
        NDArray[np.int16]: Array of integer scores corresponding to win rates.
    """
    scores = np.empty_like(win_rate, dtype=np.int16)
    scores[win_rate == 1.0] = 32000
    scores[win_rate == 0.0] = -32000
    mask = (win_rate != 1.0) & (win_rate != 0.0)
    scores[mask] = (-score_scaling * np.log(1.0 / win_rate[mask] - 1)).astype(np.int16)

    return scores


class BatchBuffer:
    def __init__(self, max_size: int, batch_size: int, dtype: DTypeLike):
        self.buffer = np.empty(max_size, dtype=dtype)
        self.start = 0
        self.end = 0
        self.max_size = max_size
        self.batch_size = batch_size

    def push(self, arr: NDArray) -> None:
        size = min(len(arr), self.max_size - self.end)
        self.buffer[self.end:self.end + size] = arr[:size]
        self.end += size

    def pop(self) -> NDArray:
        size = min(self.batch_size, self.end - self.start)
        self.start += size

        return self.buffer[self.start - size:self.start]

    def empty(self) -> bool:
        return self.start == self.end


class DuplicatorChecker:
    def __init__(self, capacity=2 ** 34):
        self.filter = np.zeros(capacity // 64, dtype=np.uint64)
        self.capacity = capacity

    def _hash_index(self, hash_val: np.uint64) -> tuple[int, int]:
        index = hash_val % self.capacity

        return index // 64, index % 64

    def check(self, hash_val: np.uint64) -> bool:
        array_pos, bit_pos = self._hash_index(hash_val)
        mask = np.uint64(1 << bit_pos)

        return self.filter[array_pos] & mask

    def mark(self, hash_val: np.uint64) -> bool:
        array_pos, bit_pos = self._hash_index(hash_val)
        mask = np.uint64(1 << bit_pos)
        self.filter[array_pos] |= mask
