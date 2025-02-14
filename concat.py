import argparse
from glob import glob
import os
from typing import List, Type

from cshogi import PackedSfenValue
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate multiple PSV files into a single file.")
    parser.add_argument("files", nargs='+', help="List of PSV files or a directory containing them")
    parser.add_argument("output_file", type=str, help="Output file (.bin)")
    parser.add_argument("--chunk-size", type=int, default=10**7, help="Chunk size", dest="chunk_size")

    return parser.parse_args()


def concat_files(files: List[str], output_path: str, chunk_size: int, dtype: Type) -> None:
    type_size = np.dtype(dtype).itemsize
    total = sum(os.path.getsize(file) for file in files) // type_size
    output_pos = 0
    output_mmap = np.memmap(output_path, dtype=dtype, mode="w+", shape=(total,))

    with tqdm(total=total, desc="Concatenating files") as bar:
        for file in files:
            file_size = os.path.getsize(file)

            if file_size % type_size != 0:
                print(f"File {file} is broken (not a multiple of {PackedSfenValue.itemsize}).\n"
                       "Skipping.")
                continue

            num_elem = file_size // type_size
            processed = 0
            input_mmap = np.memmap(file, dtype=dtype, mode="r")

            while processed < num_elem:
                end = min(processed + chunk_size, num_elem)
                current_chunk = end - processed
                output_mmap[output_pos:output_pos+current_chunk] = input_mmap[processed:processed+current_chunk]

                # Update progress bar
                bar.update(current_chunk)

                processed += current_chunk
                output_pos += current_chunk

    output_mmap.flush()


def verify_files(files: List[str], dtype: Type) -> List[str]:
    verified_files = []

    for file in files:
        if not file.endswith(".bin"):
            print("Skipping file", file)
            continue

        if os.path.getsize(file) % np.dtype(dtype).itemsize != 0:
            print(f"Input file {file} is broken (not a multiple of {PackedSfenValue.itemsize}).")
            print("Skipping.")
            continue

        verified_files.append(file)

    return verified_files


def main() -> None:
    args = parse_args()
    dtype=PackedSfenValue

    if not args.output_file.endswith(".bin"):
        raise ValueError("Output file must be a PackedSfenValue file (.bin).")

    files_to_concat = []

    for file in args.files:
        if os.path.isfile(file):
            files_to_concat.append(file)
        elif os.path.isdir(file):
            files_to_concat.extend(glob(os.path.join(file, "*.bin")))

    verified_files = verify_files(files_to_concat, PackedSfenValue)
    concat_files(verified_files, args.output_file, chunk_size=args.chunk_size, dtype=dtype)


if __name__ == "__main__":
    main()
