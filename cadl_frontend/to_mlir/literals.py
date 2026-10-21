"""Single-round decimal literals, emitted as APFloat IEEE hexadecimal bits.
MLIR's decimal attribute parser can pass through double: do not double-round a
BF16/FP32 halfway boundary. This is compile-time constant materialization only.
"""
from decimal import Decimal
from fractions import Fraction
import circt.ir as ir


def exact_float_attr(ty: ir.Type, text: str) -> ir.Attribute:
    fmt={'bf16':(8,7),'f32':(8,23),'f64':(11,52)}
    eb,fb=fmt[str(ty)];bias=(1<<(eb-1))-1;width=1+eb+fb
    dec=Decimal(text);sign=int(dec.is_signed())<<(width-1)
    x=abs(Fraction(dec))
    if not x:bits=sign
    else:
        n,d=x.numerator,x.denominator
        e=n.bit_length()-d.bit_length()
        if (n<d<<e) if e>=0 else (n<<-e<d):e-=1
        minimum=1-bias
        scale=max(e-fb,minimum-fb)
        if scale>=0:d<<=scale
        else:n<<=-scale
        q,r=divmod(n,d)
        if 2*r>d or (2*r==d and q&1):q+=1
        if q>=(1<<(fb+1)):q>>=1;e+=1
        if e>bias:bits=sign|(((1<<eb)-1)<<fb)
        elif q<(1<<fb):bits=sign|q
        else:bits=sign|((max(e,minimum)+bias)<<fb)|(q-(1<<fb))
    return ir.Attribute.parse(f'0x{bits:0{width//4}X} : {ty}')
