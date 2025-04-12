import argparse
import os

from cshogi import PackedSfenValue
import numpy as np
from numpy.typing import DTypeLike
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shuffle data in PSV file.")
    parser.add_argument("input_file", type=str, help="Input file (.bin)")
    parser.add_argument("output_file", type=str,
                        help="Output file (.bin). If left empty, the input file will be shuffled and overwritten.")

    return parser.parse_args()


def shuffle_large_file(input_path: str,
                       output_path: str,
                       dtype: DTypeLike=PackedSfenValue) -> None:
    mmap = np.memmap(input_path, dtype=dtype, mode='r')
    indices = np.random.permutation(len(mmap))

    with open(output_path, 'wb') as f:
        for i in tqdm(indices):
            f.write(mmap[i].tobytes())


def shuffle_large_file_inplace(input_path: str,
                               dtype: DTypeLike=PackedSfenValue) -> None:
    mmap = np.memmap(input_path, dtype=dtype, mode='r+')
    indices = np.random.permutation(len(mmap))

    for i in tqdm(range(len(indices))):
        if indices[i] == i:
            continue

        temp = mmap[i].copy()
        mmap[i] = mmap[indices[i]]
        mmap[indices[i]] = temp

        indices[indices[i]] = indices[i]
        indices[i] = i

    mmap.flush()


def main() -> None:
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

        shuffle_large_file(args.input_file, args.output_file)
    else:
        shuffle_large_file_inplace(args.input_file)


if __name__ == "__main__":
    main()
