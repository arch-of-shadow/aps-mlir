# CADL (APS)

Syntax highlighting for **Computer Architecture Description Language**, the APS
language for describing custom RISC-V instruction extensions. Opens `.cadl`
files as CADL; no language server or executable extension code is required.

Highlights declarations (`flow`, `rtype`, `static`, `register`, `regfile`),
control flow, attributes (`#[opcode(...)]`), directives (`[[unroll(...)]]`),
arbitrary-width types (`u256`, `i32`, `u32cc`), floating types (`bf16`, `f32`,
`f64`), width-aware literals (`7'b0101011`, `32'shFF`), casts and floating
intrinsics, comments, strings, and processor interfaces (`_irf`, `_csr`, `_mem`,
`_burst_read`, `_burst_write`). Provides comment toggling, bracket matching,
and auto-closing pairs. Single quotes are deliberately **not** auto-closed:
they are part of width-aware numeric literals, not strings.

```cadl
#[opcode(7'b0101011)]
#[funct7(7'b0000000)]
rtype hello(rs1: u5, rs2: u5, rd: u5) {
  let a: u32 = _irf[rs1];
  let b: u32 = _irf[rs2];
  _irf[rd] = a + b;
}
```

## Install

Build from the repository:

```sh
cd editors/cadl
npm ci
npm run build:vscode
code --install-extension build/cadl-0.1.0.vsix
```

Alternatively select **Extensions → Install from VSIX…** in VS Code.

The syntax follows `cadl_frontend/grammar.lark`; highlighting does not perform
compiler validation, completion, or diagnostics. Build/test and Zed instructions
are in `editors/cadl/README.md`.
