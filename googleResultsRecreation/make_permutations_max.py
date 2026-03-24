import itertools
import random
import struct
from pathlib import Path


OUT_PATH = Path("permutations_100_max.bin")
NUM_PERMS = 1000
PERM_LEN = 9
SEED = 0
CANDIDATE_POOL_SIZE = 5000


def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))


def generate_permutations():
    random.seed(SEED)

    all_perms = list(itertools.permutations(range(PERM_LEN)))
    random.shuffle(all_perms)

    selected = [all_perms.pop()]

    while len(selected) < NUM_PERMS and all_perms:
        best_perm = None
        best_score = -1

        if len(all_perms) <= CANDIDATE_POOL_SIZE:
            candidates = all_perms
        else:
            candidates = random.sample(all_perms, CANDIDATE_POOL_SIZE)

        for cand in candidates:
            score = min(hamming_distance(cand, s) for s in selected)
            if score > best_score:
                best_score = score
                best_perm = cand

        selected.append(best_perm)
        all_perms.remove(best_perm)

        if len(selected) % 100 == 0:
            print(f"Selected {len(selected)}/{NUM_PERMS} permutations")

    return selected


def write_binary(perms, out_path):
    with open(out_path, "wb") as f:
        f.write(struct.pack("<i", len(perms)))
        f.write(struct.pack("<i", PERM_LEN))

        for perm in perms:
            for x in perm:
                f.write(struct.pack("<i", x + 1))  # 1-indexed for compatibility

    print(f"Wrote {out_path}")
    print(f"File size: {out_path.stat().st_size} bytes")


def verify_file(out_path):
    expected_size = (2 + NUM_PERMS * PERM_LEN) * 4
    print(f"Expected size: {expected_size} bytes")

    with open(out_path, "rb") as f:
        raw = f.read(8)
        count, perm_len = struct.unpack("<ii", raw)

    print(f"Header count: {count}")
    print(f"Header perm length: {perm_len}")


def main():
    perms = generate_permutations()
    write_binary(perms, OUT_PATH)
    verify_file(OUT_PATH)


if __name__ == "__main__":
    main()