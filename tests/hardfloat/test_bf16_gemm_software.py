"""Check the BF16 software reference against the exact oracle."""
import ctypes
import itertools
from pathlib import Path
import random
import subprocess
import sys

import pytest

from ieee_oracle import IEEE

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/hardfloat/bf16_gemm"


@pytest.fixture(scope="module")
def software(tmp_path_factory):
    out = tmp_path_factory.mktemp("bf16-software")
    subprocess.run([sys.executable, str(EXAMPLE / "generate_vectors.py"), str(out)], check=True)
    wrapper = out / "wrapper.c"
    wrapper.write_text('''
#include "bf16_software.h"
#include "vectors.h"
uint16_t software_add(uint16_t a, uint16_t b) { return bf16_add(a, b); }
uint16_t software_mul(uint16_t a, uint16_t b) { return bf16_mul(a, b); }
int check_gemm(void) {
    uint16_t a[512], bt[512], d[512];
    for (unsigned batch = 0; batch < 2; ++batch) {
        for (unsigned tile = 0; tile < 8; ++tile)
            for (unsigned i = 0; i < 8; ++i)
                for (unsigned k = 0; k < 8; ++k) {
                    a[i*64 + tile*8 + k] = cases[batch*8 + tile].a[i*8 + k];
                    bt[i*64 + tile*8 + k] = cases[batch*8 + tile].b[k*8 + i];
                }
        bf16_gemm_software(a, bt, d);
        for (unsigned row = 0; row < 8; ++row)
            for (unsigned col = 0; col < 64; ++col)
                if (d[row*64 + col] != cases[batch*8 + col/8].first[row*8 + col%8])
                    return 1 + batch*512 + row*64 + col;
    }
    return 0;
}
''')
    library = out / "software.so"
    subprocess.run(["cc", "-std=c11", "-O3", "-ffp-contract=off", "-fno-fast-math",
                    "-shared", "-fPIC", "-I", str(EXAMPLE), "-I", str(out),
                    str(wrapper), str(EXAMPLE / "bf16_software.c"), "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    for name in ("software_add", "software_mul"):
        function = getattr(lib, name)
        function.argtypes = [ctypes.c_uint16, ctypes.c_uint16]
        function.restype = ctypes.c_uint16
    return lib


def test_separate_rounding_matches_exact_bf16(software):
    oracle = IEEE(16)
    directed = [0, 0x8000, 1, 0x7f, 0x80, 0x81, 0x3f00, 0x3f80, 0x3f81,
                0x3f83, 0xbf80, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc0, 0x7f81]
    rng = random.Random(0xbf16c)
    pairs = list(itertools.product(directed, repeat=2))
    pairs += [(rng.getrandbits(16), rng.getrandbits(16)) for _ in range(8192)]
    for name in ("add", "mul"):
        implementation = getattr(software, "software_" + name)
        for a, b in pairs:
            expected, _ = oracle.operation(name, a, b)
            assert implementation(a, b) == expected, (name, hex(a), hex(b), hex(expected))


def test_software_gemm_matches_all_correctness_vectors(software):
    assert software.check_gemm() == 0
