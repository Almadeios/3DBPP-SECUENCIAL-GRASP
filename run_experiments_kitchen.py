import itertools
import subprocess
import sys

BUFFERS = [1, 3, 5, 10]
STEPS = [0.01, 0.02, 0.03]


def build_cmd(buffer_size, step):
    cmd = [
        sys.executable,
        "main.py",
        "--dataset", "kitchen",
        "--sequence-index", "3",
        "--restrict-rotations",
        "--irregular-order",
        "--max-passes", "2",
        "--random-seed", "42",
    ]

    if buffer_size > 1:
        cmd += [
            "--grasp-iterations", "10",
            "--rcl-size", "5",
        ]
    else:
        cmd += [
            "--grasp-iterations", "1",
            "--rcl-size", "5",
        ]

    cmd += [
        "--buffer-size", str(buffer_size),
        "--step", str(step),
    ]

    return cmd


def main():
    for buffer_size, step in itertools.product(BUFFERS, STEPS):
        cmd = build_cmd(buffer_size, step)
        print("\n>>>", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
