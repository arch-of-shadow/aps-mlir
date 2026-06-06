from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

from cadl_frontend.to_mlir import convert_cadl_to_mlir
from cadl_frontend.parser import parse_proc


def _raw_op(op):
    return op.operation if hasattr(op, "operation") else op


def _attr_map(op) -> dict[str, str]:
    return {attr.name: str(attr.attr) for attr in _raw_op(op).attributes}


def _attr_value(op, name: str) -> Optional[str]:
    attrs = _attr_map(op)
    return attrs.get(name)


@dataclass
class MLIRCheck:
    module: object

    @classmethod
    def from_cadl(cls, source: str, *, filename: str = "test.cadl") -> "MLIRCheck":
        ast = parse_proc(source, filename)
        module = convert_cadl_to_mlir(ast)
        assert module is not None
        module.operation.verify()
        return cls(module)

    @property
    def text(self) -> str:
        return str(self.module)

    def ops(self):
        out = []

        def rec(op):
            raw = _raw_op(op)
            out.append(raw)
            for region in raw.regions:
                for block in region.blocks:
                    for child in block.operations:
                        rec(child)

        rec(self.module.operation)
        return out

    def ops_named(self, name: str):
        return [op for op in self.ops() if op.name == name]

    def single_op(self, name: str):
        ops = self.ops_named(name)
        assert len(ops) == 1, f"Expected exactly one {name}, found {len(ops)}\n{self.text}"
        return ops[0]

    def op_counts(self) -> Counter:
        return Counter(op.name for op in self.ops())

    def assert_op_count(self, name: str, *, exactly: int | None = None,
                        at_least: int | None = None) -> None:
        count = self.op_counts()[name]
        if exactly is not None:
            assert count == exactly, (
                f"Expected exactly {exactly} {name} op(s), found {count}\n{self.text}"
            )
        if at_least is not None:
            assert count >= at_least, (
                f"Expected at least {at_least} {name} op(s), found {count}\n{self.text}"
            )

    def assert_no_op(self, name: str) -> None:
        self.assert_op_count(name, exactly=0)

    def func(self, sym_name: str):
        expected = f'"{sym_name}"'
        for op in self.ops_named("func.func"):
            if _attr_value(op, "sym_name") == expected:
                return op
        funcs = [_attr_value(op, "sym_name") for op in self.ops_named("func.func")]
        raise AssertionError(f"Function {sym_name} not found; functions={funcs}\n{self.text}")

    def assert_func(self, sym_name: str, *, opcode: int | None = None,
                    funct7: int | None = None,
                    function_type: str | None = None):
        op = self.func(sym_name)
        attrs = _attr_map(op)
        if opcode is not None:
            assert attrs.get("opcode") == f"{opcode} : i32", (
                f"Function {sym_name} opcode mismatch: {attrs.get('opcode')}\n{self.text}"
            )
        if funct7 is not None:
            assert attrs.get("funct7") == f"{funct7} : i32", (
                f"Function {sym_name} funct7 mismatch: {attrs.get('funct7')}\n{self.text}"
            )
        if function_type is not None:
            assert attrs.get("function_type") == function_type, (
                f"Function {sym_name} type mismatch: {attrs.get('function_type')}\n{self.text}"
            )
        return op

    def global_op(self, sym_name: str):
        expected = f'"{sym_name}"'
        for op in self.ops_named("memref.global"):
            if _attr_value(op, "sym_name") == expected:
                return op
        globals_ = [_attr_value(op, "sym_name") for op in self.ops_named("memref.global")]
        raise AssertionError(f"Global {sym_name} not found; globals={globals_}\n{self.text}")

    def assert_global(self, sym_name: str, *, type_: str | None = None,
                      attrs: dict[str, str] | None = None,
                      constant: bool | None = None):
        op = self.global_op(sym_name)
        actual = _attr_map(op)
        if type_ is not None:
            assert actual.get("type") == type_, (
                f"Global {sym_name} type mismatch: {actual.get('type')}\n{self.text}"
            )
        if constant is not None:
            has_constant = "constant" in actual
            assert has_constant is constant, (
                f"Global {sym_name} constant mismatch: {has_constant}\n{self.text}"
            )
        for key, value in (attrs or {}).items():
            assert actual.get(key) == value, (
                f"Global {sym_name} attr {key} mismatch: {actual.get(key)} != {value}\n{self.text}"
            )
        return op

    def assert_result_types(self, op_name: str, expected: list[str], *,
                            index: int = 0) -> None:
        ops = self.ops_named(op_name)
        assert len(ops) > index, f"Missing {op_name} op at index {index}\n{self.text}"
        actual = [str(result.type) for result in ops[index].results]
        assert actual == expected, (
            f"{op_name} result types mismatch: {actual} != {expected}\n{self.text}"
        )

    def assert_operand_types(self, op_name: str, expected: list[str], *,
                             index: int = 0) -> None:
        ops = self.ops_named(op_name)
        assert len(ops) > index, f"Missing {op_name} op at index {index}\n{self.text}"
        actual = [str(operand.type) for operand in ops[index].operands]
        assert actual == expected, (
            f"{op_name} operand types mismatch: {actual} != {expected}\n{self.text}"
        )

    def assert_op_attr(self, op_name: str, attr_name: str, expected: str, *,
                       index: int = 0) -> None:
        ops = self.ops_named(op_name)
        assert len(ops) > index, f"Missing {op_name} op at index {index}\n{self.text}"
        actual = _attr_value(ops[index], attr_name)
        assert actual == expected, (
            f"{op_name}.{attr_name} mismatch: {actual} != {expected}\n{self.text}"
        )

    def region_blocks(self, op):
        blocks = []
        for region in _raw_op(op).regions:
            blocks.extend(list(region.blocks))
        return blocks

    def assert_region_block_arg_types(self, op, expected: list[list[str]]) -> None:
        actual = [[str(arg.type) for arg in block.arguments]
                  for block in self.region_blocks(op)]
        assert actual == expected, (
            f"{_raw_op(op).name} region block arg types mismatch: {actual} != {expected}\n{self.text}"
        )

    def assert_region_block_ops(self, op, expected: list[list[str]]) -> None:
        actual = [[child.name for child in block.operations]
                  for block in self.region_blocks(op)]
        assert actual == expected, (
            f"{_raw_op(op).name} region block ops mismatch: {actual} != {expected}\n{self.text}"
        )

    def assert_region_terminator_operand_producers(
        self,
        op,
        expected: list[list[str | None]],
        *,
        terminator_name: str = "scf.yield",
    ) -> None:
        blocks = self.region_blocks(op)
        actual = []
        for block in blocks:
            operations = list(block.operations)
            assert operations, (
                f"{_raw_op(op).name} has an empty region block\n{self.text}"
            )
            terminator = operations[-1]
            assert terminator.name == terminator_name, (
                f"Expected region block terminator {terminator_name}, "
                f"found {terminator.name}\n{self.text}"
            )
            producers = []
            for i, _ in enumerate(terminator.operands):
                producer = self.producer_of_operand(terminator, i)
                producers.append(getattr(producer, "name", None))
            actual.append(producers)
        assert actual == expected, (
            f"{_raw_op(op).name} terminator operand producers mismatch: "
            f"{actual} != {expected}\n{self.text}"
        )

    def producer_of_operand(self, op, operand_index: int):
        operands = list(_raw_op(op).operands)
        assert len(operands) > operand_index, (
            f"{_raw_op(op).name} missing operand {operand_index}\n{self.text}"
        )
        owner = operands[operand_index].owner
        return _raw_op(owner) if hasattr(owner, "name") or hasattr(owner, "operation") else owner

    def assert_operand_producer(self, op, operand_index: int,
                                producer_name: str | None) -> None:
        producer = self.producer_of_operand(op, operand_index)
        actual = getattr(producer, "name", None)
        if producer_name is None:
            assert actual is None, (
                f"Expected operand {operand_index} of {_raw_op(op).name} to come from a block argument, "
                f"found {actual}\n{self.text}"
            )
        else:
            assert actual == producer_name, (
                f"Expected operand {operand_index} of {_raw_op(op).name} to come from {producer_name}, "
                f"found {actual}\n{self.text}"
            )

    def assert_named_operand_producer(self, op_name: str, operand_index: int,
                                      producer_name: str | None, *,
                                      index: int = 0) -> None:
        ops = self.ops_named(op_name)
        assert len(ops) > index, f"Missing {op_name} op at index {index}\n{self.text}"
        self.assert_operand_producer(ops[index], operand_index, producer_name)

    def result_users(self, op, result_index: int = 0):
        results = list(_raw_op(op).results)
        assert len(results) > result_index, (
            f"{_raw_op(op).name} missing result {result_index}\n{self.text}"
        )
        users = []
        for use in results[result_index].uses:
            owner = getattr(use, "owner", None)
            if owner is None and hasattr(use, "operation"):
                owner = use.operation
            if owner is None:
                owner = getattr(use, "operation", None)
            if owner is not None:
                users.append(_raw_op(owner))
        return users

    def assert_result_has_user(self, op, user_name: str, *,
                               result_index: int = 0) -> None:
        users = self.result_users(op, result_index)
        names = [getattr(user, "name", None) for user in users]
        assert user_name in names, (
            f"Expected result {result_index} of {_raw_op(op).name} to be used by {user_name}, "
            f"users={names}\n{self.text}"
        )
