import ctypes
import random
from pathlib import Path
import pytest
from ieee_oracle import IEEE

class F32(ctypes.Structure):
    _fields_=[('bits',ctypes.c_uint32)]


class BF16(ctypes.Structure):
    _fields_=[('bits',ctypes.c_uint16)]


@pytest.mark.parametrize('width',[16,32])
def test_exact_oracle_against_softfloat(width):
    root=Path(__file__).resolve().parents[2]
    sf=ctypes.CDLL(str(root/'.pixi/envs/default/riscv-tools/lib/libsoftfloat.so'))
    rm=ctypes.c_uint8.in_dll(sf,'softfloat_roundingMode')
    tiny=ctypes.c_uint8.in_dll(sf,'softfloat_detectTininess'); tiny.value=1
    ef=ctypes.c_uint8.in_dll(sf,'softfloat_exceptionFlags')
    oracle=IEEE(width); rng=random.Random(20261004)
    special=[0,oracle.signbit,1,(1<<oracle.f)-1,1<<oracle.f,127<<oracle.f,(127<<oracle.f)|oracle.signbit,oracle.max,oracle.max|oracle.signbit,oracle.inf,oracle.inf|oracle.signbit,oracle.nan,oracle.inf|1]
    scalar=BF16 if width==16 else F32
    rows=[(a,b,special[(i+j)%len(special)]) for i,a in enumerate(special) for j,b in enumerate(special)]
    # Tininess is checked after precision rounding with unbounded exponent,
    # not by checking the final output's exponent field. RUP can return the
    # minimum normal while raising UF|NX for these exact fused results.
    rows += [(1,1,(1<<oracle.f)-1),(oracle.signbit|1,1,oracle.signbit|((1<<oracle.f)-1))]
    rows += [tuple(rng.getrandbits(width) for _ in range(3)) for i in range(500)]
    for name in ('add','sub','mul','div','sqrt','fma'):
        func=getattr(sf,('bf16_' if width==16 else 'f32_')+('mulAdd' if name=='fma' else name))
        n=1 if name=='sqrt' else 3 if name=='fma' else 2
        func.argtypes=[scalar]*n; func.restype=scalar
        for rounding in range(5):
            rm.value=rounding
            for a,b,c in rows:
                ef.value=0
                result=func(*[scalar(v) for v in (a,b,c)[:n]]).bits
                if oracle.decode(result)[0]=='nan':result=oracle.nan
                expected=oracle.operation(name,a,b,c,rm=rounding)
                assert (result,ef.value)==expected,(name,rounding,hex(a),hex(b),hex(c),(hex(result),ef.value),expected)
