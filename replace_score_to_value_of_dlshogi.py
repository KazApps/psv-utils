# This script is based on https://github.com/tayayan/cshogi_util/blob/main/psvscore_to_dlvalue.py

import argparse
import math
import os

from cshogi import Board, PackedSfenValue
from cshogi.dlshogi import make_input_features, FEATURES1_NUM, FEATURES2_NUM
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Replace score in psv file with Value in dlshogi.")
    parser.add_argument("input_file", type=str, help="Input file (.bin)")
    parser.add_argument("output_file", type=str, help="Output file (.bin)")
    parser.add_argument("--model-path", type=str, default="model.onnx", help="ONNX Model path", dest="model_path")
    parser.add_argument("--score-scaling", type=float, default=600.0, help="Score scaling", dest="score_scaling")
    parser.add_argument("--batch-size", type=int, default=16384, help="Batch size", dest="batch_size")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID to be used", dest="device_id")
    parser.add_argument("--enable-cuda", action="store_true", help="Enable CUDAExecutionProvider", dest="enable_cuda")
    parser.add_argument("--enable-tensorrt", action="store_true", help="Enable TensorrtExecutionProvider", dest="enable_tensorrt")
    parser.add_argument("--resume", action="store_true", help="Resume from the middle of the file", dest="resume")
    return parser.parse_args()

def setup_session(model_path, device_id, enable_cuda, enable_tensorrt):
    available_providers = ort.get_available_providers()
    providers = []
    if enable_tensorrt and 'TensorrtExecutionProvider' in available_providers:
        providers.append(('TensorrtExecutionProvider', {
            'device_id': device_id,
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
        }))
        print("TensorrtExecutionProvider is enabled.")
    if enable_cuda and 'CUDAExecutionProvider' in available_providers:
        providers.append(('CUDAExecutionProvider', {
            'device_id': device_id,
        }))
        print("CUDAExecutionProvider is enabled.")
    if 'CPUExecutionProvider' in available_providers:
        providers.append('CPUExecutionProvider')
        print("CPUExecutionProvider is enabled.")
    
    if not providers:
        raise RuntimeError("No available providers found.")

    return ort.InferenceSession(model_path, providers=providers)

def convert_to_score(win_rate, score_scaling):
    """
    Convert win_rate (float32 array) to score
    1.0 -> 32000, 0.0 -> -32000, otherwise -score_scaling * log(1/score - 1) to integer
    """
    scores = np.empty_like(win_rate, dtype=np.int16)
    scores[win_rate == 1.0] = 32000
    scores[win_rate == 0.0] = -32000
    mask = (win_rate != 1.0) & (win_rate != 0.0)
    scores[mask] = (-score_scaling * np.log(1.0 / win_rate[mask] - 1)).astype(np.int16)

    return scores

def main():
    args = parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file {args.input_file} does not exist.")
    
    if not args.input_file.endswith(".bin"):
        raise ValueError("Input file must be a PackedSfenValue file (.bin).")
    
    if not args.output_file.endswith(".bin"):
        raise ValueError("Output file must be a PackedSfenValue file (.bin).")

    # Calculate the number of all positions from the size of the input file (each position is 40 bytes)
    file_size = os.path.getsize(args.input_file)

    if file_size % PackedSfenValue.itemsize != 0:
        raise ValueError(f"Input file {args.input_file} is broken (not a multiple of {PackedSfenValue.itemsize}).")

    processed_positions = 0
    total_positions = file_size // PackedSfenValue.itemsize

    if args.resume:
        if not os.path.exists(args.output_file):
            raise FileNotFoundError(f"Output file {args.output_file} does not exist.")

        offset = os.path.getsize(args.output_file)

        if offset % PackedSfenValue.itemsize != 0:
            raise ValueError(f"Output file {args.output_file} is broken (not a multiple of {PackedSfenValue.itemsize}).")

        processed_positions = offset // PackedSfenValue.itemsize

    block_size = 10_000_000

    print("-----------------------------------------")
    print(f"Input file          : {args.input_file}")
    print(f"Number of positions : {total_positions}" + (f" (Resume from {processed_positions})" if args.resume else ""))
    print(f"Output file         : {args.output_file}")
    print(f"Model path          : {args.model_path}")
    print(f"Score scaling       : {args.score_scaling}")
    print(f"Batch size          : {args.batch_size}")
    print(f"Device ID           : {args.device_id}")
    print(f"Enable CUDA         : {args.enable_cuda}")
    print(f"Enable TensorRT     : {args.enable_tensorrt}")
    print("-----------------------------------------", end="\n\n")

    # Allocate input buffer for inference (for batch size)
    input_features1 = np.empty((args.batch_size, FEATURES1_NUM, 9, 9), dtype=np.float32)
    input_features2 = np.empty((args.batch_size, FEATURES2_NUM, 9, 9), dtype=np.float32)
    
    # Board object
    board = Board()
    
    # Setting up an ONNX Runtime session
    session = setup_session(args.model_path, args.device_id, args.enable_cuda, args.enable_tensorrt)
    print("ONNX Runtime session is ready.", end="\n\n")

    with tqdm(total=total_positions, initial=processed_positions) as bar:
        with open(args.output_file, "ab" if args.resume else "wb") as f_out:
            while processed_positions < total_positions:
                remaining_positions = total_positions - processed_positions
                read_size = min(remaining_positions, block_size)
                psvs = np.fromfile(args.input_file, count=read_size, offset=processed_positions * PackedSfenValue.itemsize, dtype=PackedSfenValue)

                if len(psvs) != read_size:
                    raise ValueError(f"Failed to read {read_size} positions from the input file.")

                win_rate = np.empty(read_size, dtype=np.float32)
                num_batches = math.ceil(read_size / args.batch_size)
                
                for b in range(num_batches):
                    start = b * args.batch_size
                    end = min(start + args.batch_size, read_size)
                    current_batch = end - start

                    # Feature creation per batch
                    for j, idx in enumerate(range(start, end)):
                        board.set_psfen(psvs[idx]["sfen"])
                        make_input_features(board, input_features1[j], input_features2[j])
                    
                    # Batch inference with IO binding
                    io_binding = session.io_binding()
                    io_binding.bind_cpu_input("input1", input_features1[:current_batch])
                    io_binding.bind_cpu_input("input2", input_features2[:current_batch])
                    io_binding.bind_output("output_policy")
                    io_binding.bind_output("output_value")
                    session.run_with_iobinding(io_binding)
                    outputs = io_binding.copy_outputs_to_cpu()
                    batch_values = outputs[1].reshape(-1)
                    win_rate[start:end] = batch_values
                    
                    # Update progress bar
                    bar.update(current_batch)
                    
                # Convert win rate to score
                scores = convert_to_score(win_rate, args.score_scaling)
                psvs["score"] = scores

                # Append to output file
                psvs.tofile(f_out)

                # Update processed positions
                processed_positions += read_size

    print(f"The score replacement process is complete. Output file : {args.output_file}")

if __name__ == "__main__":
    main()
