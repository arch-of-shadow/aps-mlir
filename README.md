# CADL Frontend (Python)

A Python implementation of the CADL (Computer Architecture Description Language) parser using Lark.

This project is the Python equivalent of the Rust `cadl_rust` parser, providing the same functionality with Python bindings.

## Features

- Full CADL language parsing
- AST generation matching the Rust implementation
- Expression evaluation and type checking
- Support for flows, functions, statements, and more

## Installation

```bash
pip install -e .
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Usage

```python
from cadl_frontend import parse_proc

# Parse CADL source code
with open("example.cadl", "r") as f:
    source = f.read()

ast = parse_proc(source, "example.cadl")
print(ast)
```