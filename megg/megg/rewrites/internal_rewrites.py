"""Internal rewrite rules for common mathematical operations using egglog."""

import logging
from collections.abc import Iterable

import egglog
from egglog import *

from megg.egraph.term import LitTerm, Term, eclass_ty

logger = logging.getLogger(__name__)
def basic_math_laws() -> egglog.Ruleset:
    """Get essential mathematical rewrite rules that are safe and practical."""
    i32_ty = egglog.String('i32')
    index_ty = egglog.String('index')

    a, b, c, d = vars_("a b c d", Term)

    rules = []

    # For matching to work, the Term structure must exactly match the pattern,

    rules.append(
        rewrite(
            Term.shrsi(
                Term.add(a, b, i32_ty),
                Term.lit(LitTerm.int(egglog.i64(1)), i32_ty),
                i32_ty
            )
        ).to(
            Term.add(
                Term.and_(a, b, i32_ty),
                Term.shrsi(
                    Term.xor_(a, b, i32_ty),
                    Term.lit(LitTerm.int(egglog.i64(1)), i32_ty),
                    i32_ty
                ),
                i32_ty
            )
        )
    )

    for ty in [i32_ty, index_ty]:

        rules.append(
            rewrite(
                Term.add(a, b, ty)
            ).to(
                Term.add(b, a, ty)
            )
        )
        rules.append(
            rewrite(
                Term.mul(a, b, ty)
            ).to(
                Term.mul(b, a, ty)
            )
        )
        rules.append(
            rewrite(
                Term.add(Term.add(a, b, ty), c, ty)
            ).to(
                Term.add(a, Term.add(b, c, ty), ty)
            )
        )
        rules.append(
            rewrite(
                Term.mul(Term.mul(a, b, ty), c, ty)
            ).to(
                Term.mul(a, Term.mul(b, c, ty), ty)
            )
        )
    
        rules.append(
            rewrite(
                Term.add(Term.mul(a, c, ty), Term.mul(c, b, ty), ty)
            ).to(
                Term.mul(Term.add(a, b, ty), c, ty)
            )
        )

        rules.append(
            rewrite(
                Term.add(Term.mul(c, a, ty), Term.mul(b, c, ty), ty)
            ).to(
                Term.mul(Term.add(a, b, ty), c, ty)
            )
        )

        rules.append(
            rewrite(
                Term.add(Term.mul(a, c, ty), Term.mul(b, c, ty), ty)
            ).to(
                Term.mul(Term.add(a, b, ty), c, ty)
            )
        )

        rules.append(
            rewrite(
                Term.add(Term.mul(c, a, ty), Term.mul(c, b, ty), ty)
            ).to(
                Term.mul(c, Term.add(a, b, ty), ty)
            )
        )
        
        rules.append(
            rewrite(
                Term.sub(
                    Term.add(
                        Term.mul(a, a, ty),
                        Term.mul(b, b, ty),
                        ty
                    ),
                    Term.mul(
                        Term.lit(LitTerm.int(egglog.i64(2)), ty),
                        Term.mul(a, b, ty),
                        ty
                    ),
                    ty
                )
            ).to(
                Term.mul(
                    Term.sub(a, b, ty),
                    Term.sub(a, b, ty),
                    ty
                )
            )
        )
        
        rules.append(
            rewrite(
                Term.sub(
                    Term.mul(a, c, ty),
                    Term.mul(b, c, ty),
                    ty
                )
            ).to(
                Term.mul(
                    Term.sub(a, b, ty),
                    c,
                    ty
                )
            )
        )

        rules.append(
            rewrite(
                Term.add(a, Term.lit(LitTerm.int(egglog.i64(0)), ty), ty)
            ).to(a)
        )

        rules.append(
            rewrite(
                Term.mul(a, Term.lit(LitTerm.int(egglog.i64(1)), ty), ty)
            ).to(a)
        )

        rules.append(
            rewrite(
                Term.mul(a, Term.lit(LitTerm.int(egglog.i64(0)), ty), ty)
            ).to(Term.lit(LitTerm.int(egglog.i64(0)), ty))
        )

        rules.append(
            rewrite(Term.neg(Term.neg(a, ty), ty)).to(a)
        )

        rules.append(
            rewrite(Term.sub(a, b, ty)).to(Term.add(a, Term.neg(b, ty), ty))
        )

        rules.append(
            rewrite(Term.add(a, Term.neg(b, ty), ty)).to(Term.sub(a, b, ty))
        )

        rules.append(
            rewrite(Term.sub(a, a, ty)).to(Term.lit(LitTerm.int(egglog.i64(0)), ty))
        )

        rules.append(
            rewrite(Term.add(a, Term.neg(a, ty), ty)).to(Term.lit(LitTerm.int(egglog.i64(0)), ty))
        )

        rules.append(
            rewrite(Term.add(Term.neg(a, ty), a, ty)).to(Term.lit(LitTerm.int(egglog.i64(0)), ty))
        )

        if ty == i32_ty:
            (ci,) = vars_("ci", egglog.i64)
            const_ci = Term.lit(LitTerm.int(ci), ty)

            rules.append(
                rewrite(
                    Term.div(
                        Term.mul(a, const_ci, ty),
                        const_ci,
                        ty,
                    )
                ).to(a)
            )

            rules.append(
                rewrite(
                    Term.div(
                        Term.mul(const_ci, a, ty),
                        const_ci,
                        ty,
                    )
                ).to(a)
            )

            for n in range(1, 5):  # Support shifts from 1 to 4 bits
                power_of_2 = 1 << n  # Calculate 2^n
                rules.append(
                    rewrite(
                        Term.shl(a, Term.lit(LitTerm.int(egglog.i64(n)), ty), ty)
                    ).to(
                        Term.mul(a, Term.lit(LitTerm.int(egglog.i64(power_of_2)), ty), ty)
                    )
                )

    basic_math_laws_min = egglog.ruleset(*rules, name='basic_math_laws_min')

    ops = [Term.add, Term.mul, Term.sub, Term.div]

    @basic_math_laws_min.register
    def _pushdown_index_cast(x: Term, y: Term, ai: i64) -> Iterable[RewriteOrRule]:
        
        for op in ops:
            yield rewrite(
                Term.index_cast(op(x, y, i32_ty), index_ty)
            ).to(
                op(Term.index_cast(x, index_ty), Term.index_cast(y, index_ty), index_ty),
                eclass_ty(x) == i32_ty, eclass_ty(y) == i32_ty
            )
            yield rewrite(
                Term.index_cast(op(x, y, index_ty), i32_ty)
            ).to(
                op(Term.index_cast(x, i32_ty), Term.index_cast(y, i32_ty), i32_ty),
                eclass_ty(x) == index_ty, eclass_ty(y) == index_ty
            )
            
            yield rewrite(
                Term.index_cast(op(x, y, i32_ty), index_ty)
            ).to(
                op(Term.index_cast(x, index_ty), y, index_ty),
                eclass_ty(x) == i32_ty, eclass_ty(y) == index_ty
            )
            yield rewrite(
                Term.index_cast(op(x, y, index_ty), i32_ty)
            ).to(
                op(Term.index_cast(x, i32_ty), y, i32_ty),
                eclass_ty(x) == index_ty, eclass_ty(y) == i32_ty
            )
        
        yield rewrite(
            Term.index_cast(Term.lit(LitTerm.int(ai), i32_ty), index_ty)
        ).to(
            Term.lit(LitTerm.int(ai), index_ty)
        )
        yield rewrite(
            Term.index_cast(Term.lit(LitTerm.int(ai), index_ty), i32_ty)
        ).to(
            Term.lit(LitTerm.int(ai), i32_ty)
        )
        
        yield rewrite(
            Term.index_cast(Term.loop_index(ai, i32_ty), index_ty)
        ).to(
            Term.loop_index(ai, index_ty)
        )
        yield rewrite(
            Term.index_cast(Term.loop_index(ai, index_ty), i32_ty)
        ).to(
            Term.loop_index(ai, i32_ty)
        )
        
        yield rewrite(
            Term.index_cast(Term.index_cast(x, index_ty), i32_ty)
        ).to(x,
            eclass_ty(x) == i32_ty
        )
        
        yield rewrite(
            Term.index_cast(Term.index_cast(x, i32_ty), index_ty)
        ).to(x,
            eclass_ty(x) == index_ty
        )

    return basic_math_laws_min

def type_annotation_ruleset() -> egglog.Ruleset:
        """Create type annotation rules for e-graph terms."""
        e, x, y, z = egglog.vars_("e x y z", Term)
        t = egglog.var("t", egglog.String)  # type
        i = egglog.var("i", egglog.i64)
        l = egglog.var("l", LitTerm)
        stmts = egglog.var("stmts", egglog.Vec[Term])

        type_rules = [
            (Term.lit(l, t), []),
            (Term.arg(i, t), []),
            (Term.loop_index(i, t), []),
            (Term.loop_iter_arg(i, i, t), []),
            (Term.block_arg(i, i, t), []),
        ]

        binary_ops = [
            "add",    # add (handles both addi and addf)
            "sub",    # subtract (handles both subi and subf)
            "mul",    # multiply (handles both muli and mulf)
            "div",    # divide (handles both divi and divf)
            "rem",    # remainder
            "shl",    # shift left
            "shrsi",  # shift right signed
            "shrui",  # shift right unsigned
        ]

        unary_ops = [
            "neg",    # negate
        ]

        cast_ops = [
            "index_cast",  # index to/from integer
            "sitofp",      # signed int to float
            "uitofp",      # unsigned int to float
            "fptosi",      # float to signed int
            "fptoui",      # float to unsigned int
            "extsi",       # sign extend
            "extui",       # zero extend
            "trunci",      # truncate integer
            "bitcast",     # bitcast
        ]

        for op in binary_ops:
            type_rules.append((getattr(Term, op)(x, y, t), []))

        for op in unary_ops:
            type_rules.append((getattr(Term, op)(x, t), []))

        for op in cast_ops:
            type_rules.append((getattr(Term, op)(x, t), []))

        pred = egglog.var("pred", egglog.String)
        type_rules.append((Term.cmp(x, y, pred, t), []))

        rules = [egglog.rule(egglog.eq(e).to(term_pattern)).then(
            egglog.set_(eclass_ty(e)).to(t)) for term_pattern, _ in type_rules]

        return egglog.ruleset(*rules, name='type_annotation')

def constant_folding_laws() -> egglog.Ruleset:
    """修正后的常量折叠 ruleset。"""
    i32_ty = egglog.String("i32")
    f64_ty = egglog.String("f64")

    ai, bi = vars_("ai bi", egglog.i64)
    af, bf = vars_("af bf", egglog.f64)

    rules = []

    # every addition into a subtraction. Negation is modulo the operand width.
    a = egglog.var("add_lhs", Term)
    for width in (8, 16, 32):
        integer_ty = egglog.String(f"i{width}")
        sign_bit = egglog.i64(1 << (width - 1))
        mask = egglog.i64((1 << width) - 1)
        negative = (((egglog.i64(0) - (ai & mask)) + sign_bit) & mask) - sign_bit
        rules.append(
            rewrite(Term.add(a, Term.lit(LitTerm.int(ai), integer_ty), integer_ty)).to(
                Term.sub(a, Term.lit(LitTerm.int(negative), integer_ty), integer_ty)
            )
        )

    rules.append(
        rewrite(
            Term.add(Term.lit(LitTerm.int(ai), i32_ty), Term.lit(LitTerm.int(bi), i32_ty), i32_ty)
        ).to(
            Term.lit(LitTerm.int(ai + bi), i32_ty)
        )
    )

    rules.append(
        rewrite(
            Term.sub(Term.lit(LitTerm.int(ai), i32_ty), Term.lit(LitTerm.int(bi), i32_ty), i32_ty)
        ).to(
            Term.lit(LitTerm.int(ai - bi), i32_ty)
        )
    )

    rules.append(
        rewrite(
            Term.mul(Term.lit(LitTerm.int(ai), i32_ty), Term.lit(LitTerm.int(bi), i32_ty), i32_ty)
        ).to(
            Term.lit(LitTerm.int(ai * bi), i32_ty)
        )
    )

    rules.append(
        rewrite(
            Term.neg(Term.lit(LitTerm.int(ai), i32_ty), i32_ty)
        ).to(
            Term.sub(
                Term.lit(LitTerm.int(egglog.i64(0)), i32_ty),
                Term.lit(LitTerm.int(ai), i32_ty),
                i32_ty
            )
        )
    )

    rules.append(
        rewrite(
            Term.add(Term.lit(LitTerm.float(af), f64_ty), Term.lit(LitTerm.float(bf), f64_ty), f64_ty)
        ).to(
            Term.lit(LitTerm.float(af + bf), f64_ty)
        )
    )

    rules.append(
        rewrite(
            Term.mul(Term.lit(LitTerm.float(af), f64_ty), Term.lit(LitTerm.float(bf), f64_ty), f64_ty)
        ).to(
            Term.lit(LitTerm.float(af * bf), f64_ty)
        )
    )

    rules.append(
        rewrite(
            Term.neg(Term.lit(LitTerm.float(af), f64_ty), f64_ty)
        ).to(
            Term.sub(
                Term.lit(LitTerm.float(egglog.f64(0.0)), f64_ty),
                Term.lit(LitTerm.float(af), f64_ty),
                f64_ty
            )
        )
    )

    return egglog.ruleset(*rules, name="constant_folding_laws")

def register_internal_rewrites(egraph: EGraph, include_constant_folding: bool = False) -> int:
    """Register all internal mathematical rewrite rules with the given e-graph."""
    rule_cnt = 0

    math_laws = basic_math_laws()
    egraph.run(math_laws.saturate())
    rule_cnt += len(math_laws.__egg_ruleset__.rules)

    if include_constant_folding:
        constant_rules = constant_folding_laws()
        rule_cnt += len(constant_rules.__egg_ruleset__.rules)
        egraph.run(constant_rules.saturate())
        logger.info("Registered basic mathematical laws + constant folding rules")
    else:
        logger.info("Registered basic mathematical laws")

    return rule_cnt

__all__ = [
    'basic_math_laws',
    'constant_folding_laws',
    'type_annotation_ruleset',
    'register_internal_rewrites',
]
