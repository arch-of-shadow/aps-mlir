1) lerp —— 线性插值的等价规范化

目标：验证 e-graph 对分配/结合、操作数重排与跨式子 CSE 的统一与抽取（偏好“mul+add”成对，便于整数 MAC）。

输入：%a: i32, %b: i32, %t: i32（若需 1，用 arith.constant 1 : i32）。

A_exact.mlir（差值式）：r = a + t*(b - a)
片段：%diff = arith.subi %b, %a : i32; %scaled = arith.muli %diff, %t; %r = arith.addi %a, %scaled。

B_equiv.mlir（凸组合式）：r = (1 - t)*a + t*b
片段：%one = arith.constant 1 : i32; %one_t = arith.subi %one, %t; %ta = arith.muli %one_t, %a; %tb = arith.muli %t, %b; %r = arith.addi %ta, %tb。

C_cse.mlir（共享子表达式变体）：构造 两个 输出：

r1 = a + t*(b - a) 与 r2 = (1 - t)*a + t*b 出自同一模块，显式共享 %t、%a、%b，并允许复用 %diff 或 %one_t；不使用控制流。

期望：e-graph 在 A/B/C 之间做等价饱和，能抽取到乘加相邻（MAC 友好）的形态；CSE 合并 %t、%diff/%one_t 等。

2) dot3 —— 三项点积的 AC 归一 + 乘加链化

目标：验证对加法结合/交换的 AC 归一、乘项交换，以及跨式子 CSE；抽取“(mul+add) 成对”的累加形态。

输入：%a0,%a1,%a2,%b0,%b1,%b2: i32。

A_exact.mlir（左结合）：

%p0 = muli a0,b0; %p1 = muli a1,b1; %p2 = muli a2,b2; %s01 = addi p0,p1; %r = addi s01,p2。

B_equiv.mlir（重排+交换）：

任意不同的加法树（如右结合）和乘项次序交换（muli b?,a?），保持等价求和。

C_cse.mlir（共享乘积）：

生成 两个 结果：r = dot3(a,b) 与 u = p0 + something，其中 p0 = muli a0,b0 被 复用；或构造 r、u 为两个不同加法树，但共享至少一个 %pi。

期望：e-graph 通过 AC 归一把不同加法树合并，并优先抽取“乘后紧跟加”的链式结构；CSE 保留共享的 %pi。

3) horner3 —— 三次多项式的 Horner 化

目标：展示全局重结合带来的乘法树高度降低与更规整的 mul→add→mul→add 流水。

输入：%x,%c0,%c1,%c2,%c3: i32（系数常量亦可用 arith.constant）。

A_exact.mlir（朴素展开）：

%x2 = muli x,x; %x3 = muli x2,x; r = c0 + c1*x + c2*x2 + c3*x3（用 muli/addi 逐步实现）。

B_equiv.mlir（Horner 形）：

r = (((c3*x)+c2)*x + c1)*x + c0。

C_cse.mlir（共享子表达式）：

构造 两个 输出 r 与 s，使其 共享 %x2（和/或 %x3），例如 s = r + k*x（k 为常量或参数），要求在文件内显式复用 %x2/%x3 以触发 CSE。

期望：e-graph 在 A/B/C 之间做等价饱和，抽取乘法树高度更低、乘加相邻的实现；合并 %x2/%x3 的共享。

4) clamp —— cmp+select 归约为 min/max 的规范化

目标：把两级 cmp+select 收敛到 minsi(maxsi(x,lo),hi)，并通过谓词/参数规范化扩大匹配；展示 min/max 形态利于后端映射与去分支。

输入：%x,%lo,%hi: i32。

A_exact.mlir（两级 select）：

%lt = cmpi slt, x, lo; %xlo = select lt, lo, x; %gt = cmpi sgt, xlo, hi; %r = select gt, hi, xlo。

B_equiv.mlir（min/max 语义算子）：

%r = minsi( maxsi(x, lo), hi )（仅 arith）。

C_cse.mlir（共享中间值）：

在同一模块中计算 两个 clamp 结果 r1、r2，对相同的 x,lo,hi；构造方式使 maxsi(x,lo)（或对应 select 中间值）可被复用，以检验 CSE（例如先算 xlo = maxsi(x,lo)，再分别与不同 hi1/hi2 结合，或两次相同 hi 也可）。

期望：e-graph 把 A/B 统一到 min/max 规范形态，并合并共享中间值；统计 cmp+select → min/max 的收敛率。