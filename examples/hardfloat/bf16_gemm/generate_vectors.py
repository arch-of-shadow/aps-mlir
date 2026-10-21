#!/usr/bin/env python3
"""Exact goldens for the partitioned/unrolled BF16 version of features/gemm."""
import argparse
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests/hardfloat"))
from ieee_oracle import IEEE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0xBF168008)
    oracle = IEEE(16)

    def finite():
        return (rng.randrange(2) << 15) | (rng.randrange(123, 132) << 7) | rng.randrange(128)

    identity = [0x3F80 if i == j else 0 for i in range(8) for j in range(8)]
    ordinary = [finite() for _ in range(64)]
    cases = [([0] * 64, ordinary), (identity, ordinary), (ordinary, identity)]
    for pool in ([0x3F80, 0xBF80, 0x3F00, 0xBF00],
                 [0x3F81, 0x3F83, 0x3F01, 0xBF81],
                 [0x0001, 0x007F, 0x0080, 0x8080],
                 [0x7F7F, 0xFF7F, 0x4000, 0x3F00],
                 [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x7F81]):
        cases.append(([rng.choice(pool) for _ in range(64)],
                      [rng.choice(pool) for _ in range(64)]))
    for test in range(8):
        generate = finite if test < 4 else lambda: rng.getrandbits(16)
        cases.append(([generate() for _ in range(64)], [generate() for _ in range(64)]))

    records = []
    for a, b in cases:
        first, accumulated = [], []
        for i in range(8):
            for j in range(8):
                acc = 0
                for k in range(8):
                    product, _ = oracle.operation("mul", a[i * 8 + k], b[k * 8 + j])
                    acc, _ = oracle.operation("add", acc, product)
                result, _ = oracle.operation("add", acc, 0)
                first.append(result)
                result, _ = oracle.operation("add", acc, result)
                accumulated.append(result)
        matrices = ",\n".join("        {" + ", ".join(f"0x{bits:04x}" for bits in matrix) + "}"
                              for matrix in (a, b, first, accumulated))
        records.append("    {\n" + matrices + "\n    }")
    (args.output / "vectors.h").write_text(
        "/* Generated exact BF16 multiply/add RNE goldens, B in normal order. */\n"
        "#include <stdint.h>\n"
        "struct gemm_case { uint16_t a[64], b[64], first[64], accumulated[64]; };\n"
        "static const struct gemm_case cases[] = {\n" + ",\n".join(records) + "\n};\n")
    print(f"Generated {len(cases)} 8x8 inputs, two passes each (2048 outputs).")


if __name__ == "__main__":
    main()
