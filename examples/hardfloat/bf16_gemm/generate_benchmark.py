#!/usr/bin/env python3
"""Eight dense, finite 8x8 BF16 tiles and independently rounded goldens."""
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
    rng = random.Random(0xBF1680B)
    oracle = IEEE(16)
    fp32 = IEEE(32)
    fp32_expected = [0] * 512
    # Mixed signs and nontrivial mantissas; avoid exceptional-value fast paths.
    def value():
        return (rng.randrange(2) << 15) | (rng.randrange(124, 130) << 7) | rng.randrange(128)
    a, bt, expected = ([0] * 512 for _ in range(3))
    for tile in range(8):
        for row in range(8):
            for col in range(8):
                index = row * 64 + tile * 8 + col
                a[index], bt[index] = value(), value()
        for i in range(8):
            for j in range(8):
                acc = 0
                for k in range(8):
                    product, _ = oracle.operation("mul", a[i * 64 + tile * 8 + k],
                                                   bt[j * 64 + tile * 8 + k])
                    acc, _ = oracle.operation("add", acc, product)
                expected[i * 64 + tile * 8 + j], _ = oracle.operation("add", acc, 0)
                acc32 = 0
                for k in range(8):
                    acc32, _ = fp32.operation("fma", a[i * 64 + tile * 8 + k] << 16,
                                             bt[j * 64 + tile * 8 + k] << 16, acc32)
                fp32_expected[i * 64 + tile * 8 + j] = acc32
    text = "/* Deterministic seed 0xBF1680B, independent BF16 mul/add oracle. */\n"
    for name, values in (("bench_a", a), ("bench_bt", bt), ("bench_expected", expected)):
        text += f"static const uint16_t {name}[512] = {{\n"
        text += "\n".join("    " + ", ".join(f"0x{x:04x}" for x in values[i:i+8]) + ","
                          for i in range(0, 512, 8)) + "\n};\n"
    text += "static const uint32_t bench_fp32_expected[512] = {\n"
    text += "\n".join("    " + ", ".join(f"0x{x:08x}" for x in fp32_expected[i:i+8]) + ","
                      for i in range(0, 512, 8)) + "\n};\n"
    (args.output / "bench_vectors.h").write_text(text)
    print("Generated benchmark: 8 dense finite BF16 tiles / 512 outputs per trial")


if __name__ == "__main__":
    main()
