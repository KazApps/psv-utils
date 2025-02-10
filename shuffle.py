import argparse
import os

from cshogi import PackedSfenValue
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Shuffle data in PSV file.")
    parser.add_argument("input_file", type=str, help="Input file (.bin)")
    parser.add_argument("output_file", type=str,
                        help="Output file (.bin). If left empty, the input file will be shuffled and overwritten.")
    parser.add_argument("--chunk-size", type=int, default=10**7, help="Chunk size", dest="chunk_size")

    return parser.parse_args()

def shuffle_large_file(input_path, output_path, chunk_size=10**7, dtype=PackedSfenValue):
    input_mmap = np.memmap(input_path, dtype=dtype, mode='r')
    total = len(input_mmap)
    num_chunks = int(np.ceil(total / chunk_size))
    chunk_order = np.random.permutation(num_chunks)
    output_pos = 0

    if os.path.exists(output_path):
        os.remove(output_path)

    output_mmap = np.memmap(output_path, dtype=dtype, mode='w+', shape=input_mmap.shape)

    for chunk_idx in tqdm(chunk_order, desc="Shuffling chunks"):
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, total)
        chunk = input_mmap[start:end].copy()
        np.random.shuffle(chunk)
        output_end = output_pos + (end - start)
        output_mmap[output_pos:output_end] = chunk[:]
        output_pos = output_end

    assert output_pos == total, "Output data size does not match input data size"
    output_mmap.flush()

def shuffle_large_file_inplace(input_path, chunk_size=10**7, dtype=PackedSfenValue):
    mmap = np.memmap(input_path, dtype=dtype, mode="r+")
    total = len(mmap)
    num_chunks = int(np.ceil(total / chunk_size))
    rng = np.random.default_rng()

    # Shuffling chunk order with Fisher-Yates algorithm
    for i in tqdm(range(num_chunks - 1 if total % chunk_size == 0 else 2, 0, -1), desc="Shuffling chunk order"):
        j = rng.integers(0, i + 1)
        if i == j:
            continue

        start_i = i * chunk_size
        end_i = min((i + 1) * chunk_size, total)
        start_j = j * chunk_size
        end_j = min((j + 1) * chunk_size, total)

        tmp = mmap[start_i:end_i].copy()
        mmap[start_i:end_i] = mmap[start_j:end_j]
        mmap[start_j:end_j] = tmp

    for chunk_idx in tqdm(range(num_chunks), desc="Shuffling within chunks"):
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, total)
        chunk = mmap[start:end].copy()
        np.random.shuffle(chunk)
        mmap[start:end] = chunk

    mmap.flush()

def main():
    args = parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file {args.input_file} does not exist.")

    if not args.input_file.endswith(".bin"):
        raise ValueError("Input file must be a PackedSfenValue file (.bin).")

    if os.path.getsize(args.input_file) % PackedSfenValue.itemsize != 0:
        raise ValueError(f"Input file {args.input_file} is broken (not a multiple of {PackedSfenValue.itemsize}).")

    if args.output_file:
        if not args.output_file.endswith(".bin"):
            raise ValueError("Output file must be a PackedSfenValue file (.bin).")

        shuffle_large_file(args.input_file, args.output_file, args.chunk_size)
    else:
        shuffle_large_file_inplace(args.input_file, args.chunk_size)

if __name__ == "__main__":
    main()
