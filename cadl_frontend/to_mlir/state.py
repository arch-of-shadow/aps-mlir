from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

import circt.ir as ir
from .. import cadl_ast



class SymbolKind(Enum):
    VALUE = "value"
    GLOBAL = "global"
    CONSTANT = "constant"


@dataclass
class SymbolBinding:
    """A CADL name binding in the current MLIR conversion session."""

    value: Union[ir.Value, str, int]
    cadl_type: Optional[cadl_ast.DataType] = None
    kind: SymbolKind = SymbolKind.VALUE


# Compatibility name while converter facade is migrated.
TypedValue = SymbolBinding


@dataclass
class SymbolScope:
    """Lexical symbol scope for SSA values, globals, and constants."""

    symbols: dict[str, SymbolBinding] = field(default_factory=dict)
    parent: Optional["SymbolScope"] = None

    def get(self, name: str) -> Optional[SymbolBinding]:
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.get(name)
        return None

    def set(
        self,
        name: str,
        value: Union[ir.Value, str, int],
        cadl_type: Optional[cadl_ast.DataType] = None,
        kind: Optional[SymbolKind] = None,
    ) -> None:
        if kind is None:
            if isinstance(value, str):
                kind = SymbolKind.GLOBAL
            elif isinstance(value, int):
                kind = SymbolKind.CONSTANT
            else:
                kind = SymbolKind.VALUE
        self.symbols[name] = SymbolBinding(value, cadl_type, kind)

    def has(self, name: str) -> bool:
        return self.get(name) is not None


@dataclass
class GlobalRegistry:
    """Module-level globals and per-scope memref.get_global cache."""

    ops: dict[str, Any] = field(default_factory=dict)
    refs: dict[str, ir.Value] = field(default_factory=dict)

    def clear_refs(self) -> None:
        self.refs = {}


@dataclass
class ConversionState:
    """Mutable state shared by one CADL-to-MLIR conversion session."""

    context: ir.Context
    module: Optional[ir.Module] = None
    current_scope: SymbolScope = field(default_factory=SymbolScope)
    scope_stack: list[SymbolScope] = field(default_factory=list)
    globals: GlobalRegistry = field(default_factory=GlobalRegistry)
    constant_vars: dict[str, int] = field(default_factory=dict)
    pending_directives: list[Any] = field(default_factory=list)

    def push_scope(self) -> None:
        self.scope_stack.append(self.current_scope)
        self.current_scope = SymbolScope(parent=self.current_scope)
        self.globals.clear_refs()

    def pop_scope(self) -> None:
        if self.scope_stack:
            self.current_scope = self.scope_stack.pop()
            self.globals.clear_refs()
