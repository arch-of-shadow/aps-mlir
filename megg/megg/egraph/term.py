# mypy: disable-error-code="empty-body,"
from __future__ import annotations
import egglog
from egglog import StringLike, eq, EGraph, rewrite, vars_, Vec

class LitTerm(egglog.Expr):
    """
    (datatype Lit
        (Int i64)
        (Float f64)
        (Complex Lit Lit)
        (APFixed i64 i64)
    )
    """
    @egglog.method(egg_fn = "Int")
    @classmethod
    def int(cls, value: egglog.i64) -> LitTerm: ...

    @egglog.method(egg_fn = "Float")
    @classmethod
    def float(cls, value: egglog.f64) -> LitTerm: ...
    # temporarily disable Complex and APFixed until we need them
    # @egglog.method(egg_fn = "APFixed")
    # @classmethod
    # def apfixed(cls, integer: egglog.i64, fractional: egglog.i64) -> LitTerm: ...



# Create the Term class with directly defined methods
class Term(egglog.Expr):
    """
    (datatype Term
        (Lit Term)
        (Arg i64)
        (<UnaryOp> Term <attr>*)
        (<BinaryOp> Term Term <attr>*)
    )
    """
    @egglog.method(egg_fn = "Lit")
    @classmethod
    def lit(cls, lit: LitTerm, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn = "Arg")
    @classmethod
    def arg(cls, index: egglog.i64, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn = "LoopIndex")
    @classmethod
    def loop_index(cls, loop_id: egglog.i64, type: egglog.String) -> Term: ...
    
    @egglog.method(egg_fn = "LoopIterArg")
    @classmethod
    def loop_iter_arg(cls, loop_id: egglog.i64, index: egglog.i64, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn = "BlockArg")
    @classmethod
    def block_arg(cls, block_id: egglog.i64, index: egglog.i64, type: egglog.String) -> Term: ...

    # ---------------------------
    # Unary arithmetic operations
    # ---------------------------
    @egglog.method(egg_fn="Neg")
    @classmethod
    def neg(cls, expr: Term, type: egglog.String) -> Term: ...

    # ---------------------------
    # Binary arithmetic operations
    # ---------------------------
    @egglog.method(egg_fn="Add")
    @classmethod
    def add(cls, expr: Term, other: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="Sub")
    @classmethod
    def sub(cls, expr: Term, other: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="Mul")
    @classmethod
    def mul(cls, expr: Term, other: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="Div")
    @classmethod
    def div(cls, expr: Term, other: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="Rem")
    @classmethod
    def rem(cls, expr: Term, other: Term, type: egglog.String) -> Term: ...

    # ---------------------------
    # Logical operations (bitwise)
    # ---------------------------
    @egglog.method(egg_fn="And")
    @classmethod
    def and_(cls, expr: Term, other: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="Or")
    @classmethod
    def or_(cls, expr: Term, other: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="Xor")
    @classmethod
    def xor_(cls, expr: Term, other: Term, type: egglog.String) -> Term: ...

    # ---------------------------
    # Shift operations
    # ---------------------------
    @egglog.method(egg_fn="Shl")
    @classmethod
    def shl(cls, expr: Term, other: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="ShrSI")
    @classmethod
    def shrsi(cls, expr: Term, other: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="ShrUI")
    @classmethod
    def shrui(cls, expr: Term, other: Term, type: egglog.String) -> Term: ...

    # ---------------------------
    # Cast operations
    # ---------------------------
    @egglog.method(egg_fn="IndexCast")
    @classmethod
    def index_cast(cls, expr: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="SIToFP")
    @classmethod
    def sitofp(cls, expr: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="UIToFP")
    @classmethod
    def uitofp(cls, expr: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="FPToSI")
    @classmethod
    def fptosi(cls, expr: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="FPToUI")
    @classmethod
    def fptoui(cls, expr: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="ExtSI")
    @classmethod
    def extsi(cls, expr: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="ExtUI")
    @classmethod
    def extui(cls, expr: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="TruncI")
    @classmethod
    def trunci(cls, expr: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="Bitcast")
    @classmethod
    def bitcast(cls, expr: Term, type: egglog.String) -> Term: ...

    # ---------------------------
    # Comparison operations
    # ---------------------------
    @egglog.method(egg_fn="Cmp")
    @classmethod
    def cmp(cls, lhs: Term, rhs: Term, predicate: egglog.String, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="Select")
    @classmethod
    def select(cls, cond: Term, true_value: Term, false_value: Term, type: egglog.String) -> Term: ...

    # ---------------------------
    # Control flow operations
    # ---------------------------
    @egglog.method(egg_fn="If")
    @classmethod
    def if_(cls, cond: Term, then_branch: Term, else_branch: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="While")
    @classmethod
    def while_(cls, init_value: egglog.Vec[Term] ,cond: Term, body: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="For")
    @classmethod
    def for_(cls, lower: Term, upper: Term, step: Term, index_var: Term, body: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="ForWithCarry")
    @classmethod
    def for_with_carry(cls, lower: Term, upper: Term, step: Term, index_var: Term, init_values: egglog.Vec[Term], body: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="AffineFor")
    @classmethod
    def affine_for(cls, lower_operands: egglog.Vec[Term], upper_operands: egglog.Vec[Term],
                   step: Term, index_var: Term, lower_map: egglog.String,
                   upper_map: egglog.String, body_stmts: egglog.Vec[Term], type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="AffineForWithCarry")
    @classmethod
    def affine_for_with_carry(cls, lower_operands: egglog.Vec[Term], upper_operands: egglog.Vec[Term],
                              step: Term, index_var: Term, lower_map: egglog.String,
                              upper_map: egglog.String, init_values: egglog.Vec[Term],
                              body_stmts: egglog.Vec[Term], type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="CustomInstr")
    @classmethod
    def custom_instr(cls, name: egglog.String, operands: egglog.Vec[Term], type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="ComponentInstr")
    @classmethod
    def component_instr(cls, name: egglog.String, operands: egglog.Vec[Term], type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="Block")
    @classmethod
    def block(cls, block_id: egglog.i64, stmts: egglog.Vec[Term], type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="Yield")
    @classmethod
    def yield_(cls, values: egglog.Vec[Term], type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="AffineYield")
    @classmethod
    def affine_yield(cls, values: egglog.Vec[Term], type: egglog.String) -> Term: ...

    # ---------------------------
    # Basic MemRef operations
    # ---------------------------
    @egglog.method(egg_fn="MemRefAlloc")
    @classmethod
    def alloc(cls, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="MemRefAlloca")
    @classmethod
    def alloca(cls, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="MemRefGetGlobal")
    @classmethod
    def get_global(cls, name: egglog.String, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="MemRefStore")
    @classmethod
    def store(cls, value: Term, memref: Term, index: Term, type: egglog.String) -> Term: ...

    @egglog.method(egg_fn="MemRefLoad")
    @classmethod
    def load(cls, memref: Term, index: Term, type: egglog.String) -> Term: ...

    # Return operations (for func.return)
    @egglog.method(egg_fn="Return")
    @classmethod
    def return_(cls, values: egglog.Vec[Term], type: egglog.String) -> Term: ...
    
    # Condition operations (for cond_br)
    @egglog.method(egg_fn="Condition")
    @classmethod
    def condition(cls, value: Term, iter_vars: egglog.Vec[Term], type: egglog.String) -> Term: ...


def safe_id_merge(existing, new):
    # 遇到冲突时保留前者
    return existing

@egglog.function(egg_fn="id_of", merge=safe_id_merge)
def id_of(term: Term) -> egglog.i64: ...


@egglog.function
def eclass_ty(t: Term) -> egglog.String:
    ...



if __name__ == "__main__":
    import os
    from megg.utils import get_temp_dir

    # Get temp dir (uses MEGG_TEMP_DIR env var or /tmp/megg)
    tmp_dir = str(get_temp_dir())

    # Set TMPDIR to our temp folder
    os.environ["TMPDIR"] = tmp_dir

    egraph = EGraph(save_egglog_string=True)

    a = egraph.let("A", Term.arg(egglog.i64(0), egglog.String("i32")))
    b = egraph.let("B", Term.arg(egglog.i64(1), egglog.String("i32")))
    c = egraph.let("C", Term.arg(egglog.i64(2), egglog.String("i32")))
    
    # a+b
    a_plus_b = egraph.let("a_plus_b", Term.add(a, b, egglog.String("i32")))
    # -c
    neg_c = egraph.let("neg_c", Term.neg(c, egglog.String("i32")))
    # a+b+(-c)
    a_plus_b_plus_neg_c = egraph.let("a_plus_b_plus_neg_c", Term.add(a_plus_b, neg_c, egglog.String("i32")))

    egraph.register(egglog.set_(id_of(a_plus_b_plus_neg_c)).to(egglog.i64(0)))

    x, y, z = vars_("x y z", Term)
    egraph.register(rewrite(Term.add(x, Term.neg(y, egglog.String("i32")), egglog.String("i32"))).to(Term.sub(x, y, egglog.String("i32"))))

    egraph.run(5)
    
    # Graphviz output
    graphviz = egraph._graphviz()
    output_path = os.path.join(tmp_dir, "__egraph.dot")
    graphviz.save(output_path)
    print(f"Graphviz saved to {output_path}")

    # check whether a+b+(-c) == a+b-c
    suc = egraph.check_bool(eq(a_plus_b_plus_neg_c).to(Term.sub(a_plus_b, c, egglog.String("i32"))))
    print(suc)

    print(egraph.as_egglog_string)
    
    
    from egglog import Vec

    # ----------- Vec 能正常构造和传入函数 ------------
    vec = Vec[Term](a, b, c)
    print("Created Vec:", vec)

    # 用在 Return 操作里（因为 Return(values: Vec[Term]) 是合法的）
    ret_term = egraph.let("ret_term", Term.return_(vec, egglog.String("void")))
    print("Return created:", ret_term)

    # 再注册一个恒等规则（不匹配内部元素）
    xv = egglog.var("xv", Vec[Term])
    egraph.register(rewrite(Term.return_(xv, egglog.String("void"))).to(Term.return_(xv, egglog.String("void"))))

    egraph.run(2)
    print("Vec test: OK ✅")
    
    
    
    x, y = vars_("x y", Term)

    try:
        egraph.register(
            rewrite(Term.return_(Vec[Term](x, y), egglog.String("void")))
            .to(Term.return_(Vec[Term](y, x), egglog.String("void")))
        )
        egraph.run(2)
    except Exception as e:
        print("💥 Expected failure (Ungrounded):", e)


