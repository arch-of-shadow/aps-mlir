# CADL editor extensions

Syntax highlighting for the **current APS CADL frontend**, not the unrelated
Cadl/TypeSpec API-description language. The authoritative grammar is
[`cadl_frontend/grammar.lark`](../../cadl_frontend/grammar.lark); floating
intrinsics follow `cadl_frontend/to_mlir/expr.py`.

## Build

Requirements: **Node.js 22+**, npm, Git, and a C compiler (for Tree-sitter tests).
Install/build commands run from this repository, with no APS/CIRCT rebuild:

```sh
cd editors/cadl
npm ci
npm run build
```

Outputs (ignored by Git):

- `build/cadl-0.1.0.vsix` — installable VS Code extension.
- `build/zed/` — Zed development extension.
- `build/grammar-repo/` — independent, commit-pinned local grammar repository
  used by the Zed extension. The build **does not commit the main repository**.

Individual commands: `npm test`, `npm run build:vscode`, `npm run build:zed`.
The generated Tree-sitter C parser and headers are included in source control;
regenerate them with `npm run generate` after grammar changes.

## VS Code

From `editors/cadl`:

```sh
code --install-extension build/cadl-0.1.0.vsix
```

Or use **Extensions → Install from VSIX…**, then select the package. `.cadl`
files are recognized automatically. The display name is **CADL (APS)**, language
ID `cadl`, and TextMate scope `source.cadl`. The extension has no runtime code.

## Zed

**Run the build on the machine running the Zed desktop client**, not inside a
Remote SSH terminal. The generated `file://` grammar URL refers to that machine's
filesystem; a server-side `/home/...` URL cannot be used by a different client.

1. Run `npm run build:zed` locally.
2. Open Zed's command palette and run **`zed: install dev extension`**.
3. Select the **absolute path to `editors/cadl/build/zed`**.
4. Open a `.cadl` file; its language should be **CADL**.

Zed compiles the grammar on installation. A current Zed release supporting
Tree-sitter ABI 15 is required. Zed may need network access to download its WASI
SDK; no Rust extension or language server is needed.

Zed requires a Git repository URL and revision for grammars, including local
ones. The build creates a `file://` URL to `build/grammar-repo` and pins its
snapshot commit in `build/zed/extension.toml`. Keep **both** directories in place;
this dev-extension directory is not a portable archive. After moving the repo,
changing the grammar, or updating queries, rebuild and use **`zed: rebuild
extension`** (or reinstall the dev extension).

### Remote SSH and troubleshooting

For a Remote SSH workspace, copy the `editors/cadl` **sources** to the local
client machine, excluding `build/` and `node_modules/`. Generated C sources are
already included, so preparing the local extension needs only Node.js and Git:

```sh
cd <local-copy>/editors/cadl
node scripts/build-zed.mjs
```

Install the resulting **local** `build/zed` directory, then reopen the remote
CADL file. Do not copy just the server-generated `build/zed` directory: its
manifest still references a server-side grammar repository.

`Failed to install dev extension: compiling grammar 'cadl'` is only an outer
error. Use **`zed: open log`** in the desktop client and inspect the full error
chain (Git checkout/fetch, WASI SDK download, or clang stderr). Remote server
logs do not contain the desktop client's grammar compilation errors. If both
paths are already local, report that full log and the Zed version rather than
assuming this is a grammar syntax error.

For a distributable/registry release, publish the grammar sources first and
replace the generated manifest's `[grammars.cadl]` entry with:

```toml
[grammars.cadl]
repository = "https://github.com/pku-eric-golem/aquas"
rev = "<published commit SHA containing these files>"
path = "editors/cadl/tree-sitter-cadl"
```

Do not submit the local `file://` manifest to the Zed extension registry.

## Coverage

Both editors highlight:

- Declarations, control flow, booleans, functions, strings, and both comment forms.
- Integer/floating/fixed-width types: `u1`, `i256`, `u32cc`, `usize`, `Instance`,
  `bf16`, `f32`, `f64`.
- Decimal, binary, octal, hexadecimal, signed/width-aware numeric literals,
  type suffixes, and scientific notation: `32'shFF_i32`, `0sxFF`, `1.25e-2_bf16`.
- Attributes and nested array arguments: `#[partition_dim_array([0, 1])]`.
- Directives: `[[unroll(4)]]`; indexing, slices, and burst ranges: `[0 +: ]`.
- Casts: `$signed`, `$unsigned`, `$f32`, `$f64`, `$int`, `$uint`, `bitcast<T>`.
- `bf16_*` / `fp32_*` floating intrinsics, including flag-returning variants.
- Processor interfaces: `_irf`, `_csr`, `_mem`, `_burst_read`, `_burst_write`.

Both provide comment toggling and bracket pairing. Single quotes are not paired,
since CADL uses them inside numbers. Zed additionally provides structural
indentation and a declaration outline.

These are syntax extensions, **not** compiler integrations: no LSP, diagnostics,
type checking, or completion. Tree-sitter follows the frontend's unusual operator
precedence (`&`/`&&` bind more tightly than `==`/`!=`), but deliberately permits
full expressions in slice bounds for editing/recovery. The frontend alone decides
semantic validity. `in`, `break`, and `continue` are declared keywords in Lark but
not currently statement productions; VS Code colors them as reserved words,
without adding unsupported statements to the Tree-sitter grammar. CADL strings
follow the current frontend's quote-delimited rule; no extra escape syntax or
nested block comments are invented.

## Tests and maintenance

```sh
npm test
```

- Real Oniguruma/TextMate tokenization asserts scopes, numeric formats, keyword
  synchronization with Lark, builtin names, comment/string isolation, and nested
  attribute boundaries.
- Tree-sitter corpus tests check declarations, precedence, conditionals, casts,
  directives, and slices.
- Parse the shared fixture plus all `.cadl` files under `examples/`,
  `tutorial/cadl/`, and `tests/hardfloat/`, excluding intentional `wont_pass/`
  errors. Any `ERROR` or missing syntax node fails the command.
- Compile/run all four Zed queries and assert representative highlight captures.

Relevant source files:

| Purpose | Path |
|---|---|
| VS Code registration | `vscode/package.json` |
| TextMate syntax | `vscode/syntaxes/cadl.tmLanguage.json` |
| VS Code pairing/comments | `vscode/language-configuration.json` |
| Tree-sitter grammar | `tree-sitter-cadl/grammar.js` |
| Zed manifest template | `zed/extension.toml.in` |
| Zed highlights, brackets, indentation, outline | `zed/languages/cadl/` |
| Shared example | `fixtures/highlight.cadl` |

Theme colors depend on the editor theme; no custom theme is necessary. New editor
sources are MIT-licensed (see `LICENSE`); this does not relicense the main project.
