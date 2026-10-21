#!/usr/bin/env python3
"""Bit-exact FP32 vectors and all five flags from Berkeley SoftFloat.

Set SOFTFLOAT_LIB, or use the library supplied by the Pixi RISC-V toolchain.
No Python/host floating-point computation is used as the arithmetic oracle.
"""
import ctypes
import os
import random
import sys
from pathlib import Path


class F32(ctypes.Structure):
    _fields_ = [("bits", ctypes.c_uint32)]


def generate(output):
    root = Path(__file__).resolve().parents[2]
    library = os.environ.get(
        "SOFTFLOAT_LIB",
        str(root / ".pixi/envs/default/riscv-tools/lib/libsoftfloat.so"),
    )
    softfloat = ctypes.CDLL(library)
    rounding = ctypes.c_uint8.in_dll(softfloat, "softfloat_roundingMode")
    tininess = ctypes.c_uint8.in_dll(softfloat, "softfloat_detectTininess")
    flags = ctypes.c_uint8.in_dll(softfloat, "softfloat_exceptionFlags")
    rounding.value = 0  # nearest even
    tininess.value = 1  # after rounding
    special = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x007FFFFF,
        0x00800000,
        0x00800001,
        0x3F000000,
        0x3F800000,
        0xBF800000,
        0x3F800001,
        0x33800000,
        0x337FFFFF,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC00000,
        0x7F800001,
        0xFFC12345,
    ]
    rng = random.Random(0x4846524E)
    pairs = [(a, b) for a in special for b in special]
    pairs += [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(10000)]
    output.mkdir(parents=True, exist_ok=True)
    for name in ("add", "sub", "mul"):
        function = getattr(softfloat, "f32_" + name)
        function.argtypes = [F32, F32]
        function.restype = F32
        with (output / f"{name}.txt").open("w") as stream:
            for a, b in pairs:
                flags.value = 0
                result = function(F32(a), F32(b)).bits
                if result & 0x7F800000 == 0x7F800000 and result & 0x007FFFFF:
                    result = 0x7FC00000
                stream.write(f"{a:08x} {b:08x} {result:08x} {flags.value:02x}\n")
    with (output / "chain.txt").open("w") as stream:
        for a, b in pairs:
            sum_bits = softfloat.f32_add(F32(a), F32(b)).bits
            product = softfloat.f32_mul(F32(sum_bits), F32(a)).bits
            result = softfloat.f32_sub(F32(product), F32(b)).bits
            if result & 0x7F800000 == 0x7F800000 and result & 0x007FFFFF:
                result = 0x7FC00000
            stream.write(f"{a:08x} {b:08x} {result:08x} 00\n")
    print(f"SoftFloat: {len(pairs)} vectors per operation -> {output}")


if __name__ == "__main__":
    generate(Path(sys.argv[1]))
