"""Independent exact IEEE binary32/BF16 oracle (integer/Fraction arithmetic).
No host floating operations, no HardFloat model, no double-rounded BF16 FMA.
RMs: RNE, RTZ, RDN, RUP, RMM. Flags: NV DZ OF UF NX, tininess after rounding.
"""
from fractions import Fraction as F
from math import isqrt


def pow2(e):
    return F(1 << e) if e >= 0 else F(1, 1 << -e)


class IEEE:
    def __init__(self, width):
        self.width = width
        self.f = width - 9
        self.signbit = 1 << (width - 1)
        self.inf = 255 << self.f
        self.nan = self.inf | (1 << (self.f - 1))
        self.max = self.inf - 1

    def decode(self, bits):
        sign = bool(bits & self.signbit)
        exp = (bits >> self.f) & 255
        frac = bits & ((1 << self.f) - 1)
        if exp == 255:
            return ('nan' if frac else 'inf', sign, None, bool(frac and not frac & (1 << (self.f - 1))))
        value = F(frac if exp == 0 else (1 << self.f) + frac) * pow2((exp or 1) - 127 - self.f)
        return ('finite', sign, -value if sign else value, False)

    @staticmethod
    def rounded_integer(x, sign, rm):
        q, r = divmod(x.numerator, x.denominator)
        up = (rm == 0 and (2*r > x.denominator or (2*r == x.denominator and q & 1))) or (rm == 4 and 2*r >= x.denominator) or (rm == 2 and sign and r != 0) or (rm == 3 and not sign and r != 0)
        return q + int(up), bool(r)

    def round(self, value, rm=0, zero_sign=False):
        sign = value < 0 if value else zero_sign
        if not value:
            return (self.signbit if sign else 0), 0
        x = abs(value)
        e = x.numerator.bit_length() - x.denominator.bit_length()
        if x < pow2(e): e -= 1
        # IEEE tininess-after-rounding is evaluated with unbounded exponent
        # range at the format's precision, before subnormal precision loss.
        # Thus a rounded normal result can still raise UF (e.g. RUP of
        # maxSubnormal + an extremely small positive product).
        unbounded_scale = e - self.f
        unbounded_q, _ = self.rounded_integer(x / pow2(unbounded_scale), sign, rm)
        tiny_after = F(unbounded_q) * pow2(unbounded_scale) < pow2(-126)
        scale = max(unbounded_scale, -126 - self.f)
        q, nx = self.rounded_integer(x / pow2(scale), sign, rm)
        if q >= 1 << (self.f + 1): q >>= 1; scale += 1
        e = scale + self.f
        if e > 127:
            infinity = rm in (0,4) or (rm == 2 and sign) or (rm == 3 and not sign)
            return (self.signbit if sign else 0) | (self.inf if infinity else self.max), 5
        normal = q >= 1 << self.f
        bits = ((e + 127) << self.f) | (q - (1 << self.f)) if normal else q
        return (self.signbit if sign else 0) | bits, int(nx) | (2 if nx and tiny_after else 0)

    def sqrt(self, value, rm):
        # Round exact sqrt by comparing squared rational midpoints.
        e = value.numerator.bit_length() - value.denominator.bit_length()
        if value < pow2(e): e -= 1
        scale = max(e // 2 - self.f, -126 - self.f)
        v = value / pow2(2*scale)
        q = isqrt(v.numerator // v.denominator)
        exact = F(q*q) == v
        midpoint = F((2*q+1)**2, 4)
        up = (rm==0 and (v>midpoint or v==midpoint and q&1)) or (rm==4 and v>=midpoint) or (rm==3 and not exact)
        q += int(up)
        if q >= 1 << (self.f+1): q >>= 1; scale += 1
        normal = q >= 1 << self.f
        bits = ((scale+self.f+127)<<self.f)|(q-(1<<self.f)) if normal else q
        return bits, (0 if exact else 1) | (2 if not exact and not normal else 0)

    def operation(self, name, a, b=0, c=0, rm=0, pred=0):
        da,db,dc = self.decode(a),self.decode(b),self.decode(c)
        ka,sa,va,sna=da; kb,sb,vb,snb=db; kc,sc,vc,snc=dc
        if name in ('neg','abs','copysign'):
            sign = not sa if name=='neg' else False if name=='abs' else sb
            return (a & (self.signbit-1)) | (self.signbit if sign else 0),0
        if name=='classify':
            exp=(a>>self.f)&255; frac=a&((1<<self.f)-1)
            idx = (8 if sna else 9) if ka=='nan' else (0 if sa else 7) if ka=='inf' else (3 if sa else 4) if not va else (2 if sa else 5) if exp==0 else (1 if sa else 6)
            return 1<<idx,0
        if name in ('from_i32','from_u32'):
            value=a-2**32 if name=='from_i32' and a&0x80000000 else a
            return self.round(F(value),rm)
        if name in ('to_i32','to_u32'):
            signed=name=='to_i32'; low=-(1<<31) if signed else 0; high=(1<<31)-1 if signed else (1<<32)-1
            if ka=='nan':return high,16
            if ka=='inf':return (low if sa else high)&0xffffffff,16
            q,nx=self.rounded_integer(abs(va),sa,rm); q=-q if sa else q
            if q<low or q>high:return (low if q<low else high)&0xffffffff,16
            return q&0xffffffff,int(nx)
        if name in ('widen','narrow'):
            source=IEEE(16 if name=='widen' else 32)
            target=IEEE(32 if name=='widen' else 16)
            k,s,v,sn=source.decode(a)
            if k=='nan':return target.nan,16 if sn else 0
            if k=='inf':return target.inf|(target.signbit if s else 0),0
            return target.round(v,rm,s)
        nan = ka=='nan' or kb=='nan'
        if name=='cmp':
            lt = False if nan else va<vb if ka==kb=='finite' else (sa if ka=='inf' else not sb) if ka!=kb else sa and not sb
            eq = False if nan else va==vb if ka==kb=='finite' else ka==kb and sa==sb
            gt = not nan and not lt and not eq
            table=[False,eq,gt,gt or eq,lt,lt or eq,not nan and not eq,not nan,nan,nan or eq,nan or gt,nan or gt or eq,nan or lt,nan or lt or eq,not eq,True]
            signaling=pred in (2,3,4,5,10,11,12,13)
            return int(table[pred]),16 if sna or snb or signaling and nan else 0
        if name in ('min','max'):
            if ka=='nan' and kb=='nan':result=self.nan
            elif ka=='nan':result=b
            elif kb=='nan':result=a
            elif va==vb==0:result=(a|b) if name=='min' else (a&b)
            else:
                lt=self.operation('cmp',a,b,pred=4)[0]
                result=a if (lt if name=='min' else not lt) else b
            return result,16 if sna or snb else 0
        if name=='sqrt':
            if ka=='nan':return self.nan,16 if sna else 0
            if sa and (ka=='inf' or va):return self.nan,16
            if ka=='inf':return a,0
            if not va:return a,0
            return self.sqrt(va,rm)
        sn=sna or snb or (snc if name=='fma' else False)
        if name=='sub':
            sb=not sb; vb=-vb if vb is not None else None
        if name in ('mul','fma'):
            invalid=(ka=='inf' and kb=='finite' and not vb) or (kb=='inf' and ka=='finite' and not va)
            ps=sa^sb
            product_inf=ka=='inf' or kb=='inf'
            if name=='fma':
                invalid |= product_inf and not nan and kc=='inf' and ps!=sc
            if nan or (name=='fma' and kc=='nan') or invalid:return self.nan,16 if sn or invalid else 0
            if product_inf:return self.inf|(self.signbit if ps else 0),0
            if name=='fma' and kc=='inf':return c,0
            result=va*vb
            if name=='fma':result+=vc
            zs=(ps if name=='mul' or (va*vb==0 and vc==0 and ps==sc) else rm==2)
            return self.round(result,rm,zs)
        if nan:return self.nan,16 if sn else 0
        if name=='div':
            sign=sa^sb
            if ka==kb=='inf' or ka==kb=='finite' and not va and not vb:return self.nan,16
            if ka=='inf':return self.inf|(self.signbit if sign else 0),0
            if kb=='inf':return self.signbit if sign else 0,0
            if not vb:return self.inf|(self.signbit if sign else 0),8
            return self.round(va/vb,rm,sign)
        if ka=='inf' or kb=='inf':
            if ka==kb=='inf' and sa!=sb:return self.nan,16
            sign=sa if ka=='inf' else sb
            return self.inf|(self.signbit if sign else 0),0
        return self.round(va+vb,rm,sa if not va and not vb and sa==sb else rm==2)
