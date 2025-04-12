import argparse
import os
from random import random

from cshogi import Board, PackedSfenValue, move_to, move16, move16_to_psv, BLACK, WHITE
from cshogi.dlshogi import make_input_features, make_move_label
import numpy as np
import onnxruntime as ort
import pickle as pkl
from tqdm import tqdm

import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a PSV file using the policy and value network of dlshogi.")
    parser.add_argument("output_file", type=str, help="Output file (.bin)")
    parser.add_argument("--num-positions", type=int, default=10 ** 10, help="Number of positions to generate",
                        dest="num_positions")
    parser.add_argument("--sfen-path", type=str, help="Path to the .sfen file used for setting starting positions",
                        dest="sfen_path")
    parser.add_argument("--ignore-ply", action="store_true", help="Ignore pries at the starting position",
                        dest="ignore_ply")
    parser.add_argument("--max-moves", type=int, default=100, help="Maximum number of moves from the starting position",
                        dest="max_moves")
    parser.add_argument("--policy-moves", type=int, default=3, help="Number of top moves to select from the policy network",
                        dest="policy_moves")
    parser.add_argument("--entering-king-skip-rate", type=float, default=0.7,
                        help="Probability to skip entering king positions", dest="entering_king_skip_rate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature of the policy network",
                        dest="temperature")
    parser.add_argument("--score-diff", type=int, default=300, help="Score diff", dest="score_diff")
    parser.add_argument("--score-limit", type=int, default=3000, help="Score limit", dest="score_limit")
    parser.add_argument("--smart-fen-skipping", action="store_true", help="Smart fen skipping", dest="smart_fen_skipping")
    parser.add_argument("--score-scaling", type=float, default=600.0, help="Score scaling", dest="score_scaling")
    parser.add_argument("--batch-size", type=int, default=16384, help="Batch size", dest="batch_size")
    parser.add_argument("--buffer-size", type=int, default=10 ** 8, help="Buffer size", dest="buffer_size")
    parser.add_argument("--resume", action="store_true", help="Resume from the middle of the file", dest="resume")

    parser = utils.configure_session_args(parser)

    return parser.parse_args()


def verify_output_file(output_file: str, resume: bool) -> int:
    processed_positions = 0

    if resume:
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file {output_file} does not exist.")

        file_size = os.path.getsize(output_file)
        if file_size % PackedSfenValue.itemsize != 0:
            raise ValueError(f"Output file {output_file} is corrupted.")

        processed_positions = file_size // PackedSfenValue.itemsize

    return processed_positions


def gensfen(output_path: str,
            num_positions: int,
            sfen_path: str | None,
            ignore_ply: bool,
            max_moves: int,
            policy_moves: int,
            entering_king_skip_rate: float,
            temperature: float,
            score_diff: int,
            score_limit: int,
            smart_fen_skipping: bool,
            score_scaling: float,
            batch_size: int,
            buffer_size: int,
            resume: bool,
            session: ort.InferenceSession,
            bar: tqdm) -> bool:
    board = Board()
    input_features1, input_features2 = utils.allocate_input_features(batch_size)
    generated = np.empty(batch_size * policy_moves, dtype=PackedSfenValue)
    in_checks, is_captures = np.empty(batch_size, dtype=bool), np.empty(batch_size, dtype=bool)
    duplicate_checker = utils.DuplicatorChecker()

    # Buffer
    sfens_buffer = utils.BatchBuffer(buffer_size, batch_size, dtype=PackedSfenValue)

    def get_checkpoint_path():
        save_dir = os.path.abspath(os.path.join(output_path, os.pardir))
        save_file = os.path.join(save_dir, "checkpoint.pkl")

        return save_file

    if resume:
        checkpoint_path = get_checkpoint_path()

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")

        print("Resuming from previous checkpoint.")
        print(f"Loading checkpoint from {checkpoint_path}...")

        with open(get_checkpoint_path(), "rb") as f:
            state = pkl.load(f)

        sfens_buffer = state["sfens_buffer"]
        duplicate_checker = state["duplicate_checker"]

        os.remove(checkpoint_path)
        print("Done!")

    try:
        def next_sfens(sfens, f_out):
            if len(sfens) == 0:
                return 0, np.empty(0, dtype=PackedSfenValue)

            for i, sfen in enumerate(sfens["sfen"]):
                board.set_psfen(sfen)
                make_input_features(board, input_features1[i], input_features2[i])

            batch_values, batch_logits = utils.inference(input_features1[:len(sfens)], input_features2[:len(sfens)], session)
            scores = utils.convert_to_score(batch_values, score_scaling)
            prev_scores = -(sfens["score"])

            # Filter by score and ply
            indices = ((np.abs(scores) < score_limit) & (sfens["gamePly"] <= max_moves) &
                      ((sfens["padding"] == 0) | (scores >= (prev_scores - score_diff))))

            sfens = sfens[indices]
            scores = scores[indices]
            batch_logits = batch_logits[indices]

            if num_positions < len(sfens):
                sfens = sfens[:num_positions]
                scores = scores[:num_positions]
                batch_logits = batch_logits[:num_positions]

            sfens["score"] = scores
            pos_count = 0

            for i, logits in enumerate(batch_logits):
                sfens["move"][i] = 0

                board.set_psfen(sfens["sfen"][i])
                duplicate_checker.mark(board.zobrist_hash())
                in_checks[i] = board.is_check()

                # Entering Kings are probability skipped.
                if board.king_square(BLACK) % 9 <= 2 or board.king_square(WHITE) % 9 >= 7:
                    if random() < entering_king_skip_rate:
                        continue

                legal_moves = np.array(list(board.legal_moves))

                if len(legal_moves) == 0 or board.is_nyugyoku():
                    continue

                labels = [make_move_label(move, board.turn) for move in legal_moves]
                legal_logits = logits[labels]

                bestmove = legal_moves[np.argmax(legal_logits)]
                is_captures[i] = board.piece(move_to(bestmove))
                sfens["move"][i] = move16_to_psv(move16(bestmove))
                probabilities = utils.softmax(legal_logits, temperature)
                filtered_moves = np.random.choice(legal_moves, size=min(policy_moves, len(legal_moves)), replace=False, p=probabilities)

                for move in filtered_moves:
                    board.push(move)

                    if not duplicate_checker.check(board.zobrist_hash()):
                        board.to_psfen(generated[pos_count:])
                        generated["score"][pos_count] = sfens["score"][i]
                        generated["gamePly"][pos_count] = sfens["gamePly"][i] + 1
                        generated["padding"][pos_count] = 1
                        pos_count += 1

                    board.pop()

            if smart_fen_skipping:
                sfens = sfens[~(in_checks | is_captures)[:len(sfens)]]

            sfens.tofile(f_out)

            return len(sfens), generated[:pos_count]

        # Reading starting positions
        if sfen_path:
            with open(sfen_path, 'r', encoding="utf-8-sig") as f:
                sfens = [line[5:] for line in f if line.startswith("sfen ")]
        else:
            sfens = ["lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"]

        sfens = list(set(sfens))

        if not resume:
            # `padding` is used as a startpos flag.
            psfens = np.zeros(len(sfens), dtype=PackedSfenValue)

            for i, sfen in enumerate(sfens):
                board.set_sfen(sfen)
                board.to_psfen(psfens[i:])
                psfens["gamePly"][i] = sfen.split()[4] if len(sfen.split()) > 4 else 1

            if ignore_ply:
                psfens["gamePly"] = 1

            np.random.shuffle(psfens)
            sfens_buffer.push(psfens)

        with open(output_path, "ab" if resume else "wb") as f_out:
            while num_positions:
                if sfens_buffer.empty():
                    print("\nThe generation process has stopped because buffer is empty.")

                    return False

                processed, new_sfens = next_sfens(sfens_buffer.pop(), f_out)
                np.random.shuffle(new_sfens)
                sfens_buffer.push(new_sfens)
                bar.update(processed)
                num_positions -= processed
    except KeyboardInterrupt:
        print("\nThe generation process has stopped because of KeyboardInterrupt.")
        checkpoint_path = get_checkpoint_path()
        print(f"Saving current state to {checkpoint_path}...")

        with open(checkpoint_path, "wb") as f_out:
            state = {
                "sfens_buffer": sfens_buffer,
                "duplicate_checker": duplicate_checker
            }

            pkl.dump(state, f_out)

        print("Done!")

        return False

    return True


def main() -> None:
    args = parse_args()
    processed_positions = verify_output_file(args.output_file, args.resume)

    print("-----------------------------------------")
    print(f"Number of positions     : {args.num_positions}" + (f" (Resume from {processed_positions})" if args.resume else ""))
    print(f"Output file             : {args.output_file}")
    print(f"Sfen path               : {args.sfen_path}")
    print(f"Ignore ply              : {args.ignore_ply}")
    print(f"Model path              : {args.model_path}")
    print(f"Max moves               : {args.max_moves}")
    print(f"Policy moves            : {args.policy_moves}")
    print(f"Entering king skip rate : {args.entering_king_skip_rate}")
    print(f"Temperature             : {args.temperature}")
    print(f"Score diff              : {args.score_diff}")
    print(f"Score limit             : {args.score_limit}")
    print(f"Smart fen skipping      : {args.smart_fen_skipping}")
    print(f"Score scaling           : {args.score_scaling}")
    print(f"Batch size              : {args.batch_size}")
    print(f"Buffer size             : {args.buffer_size}")
    print(f"Device ID               : {args.device_id}")
    print(f"Enable CUDA             : {args.enable_cuda}")
    print(f"Enable TensorRT         : {args.enable_tensorrt}")
    print("-----------------------------------------", end="\n\n")

    # Creating an ONNX Runtime session
    session = utils.create_session(args)
    print("ONNX Runtime session is ready.", end="\n\n")

    with tqdm(total=args.num_positions, initial=processed_positions) as bar:
        result = gensfen(args.output_file,
                         args.num_positions,
                         args.sfen_path,
                         args.ignore_ply,
                         args.max_moves,
                         args.policy_moves,
                         args.entering_king_skip_rate,
                         args.temperature,
                         args.score_diff,
                         args.score_limit,
                         args.smart_fen_skipping,
                         args.score_scaling,
                         args.batch_size,
                         args.buffer_size,
                         args.resume,
                         session,
                         bar)

    if result:
        print(f"\nGenerated {args.num_positions} positions successfully.")


if __name__ == "__main__":
    main()
