"""MeggEGraph: Python-based E-graph structure for greedy extraction."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, TypedDict

from megg.egraph.datatype import DataType
from megg.egraph.func_to_terms import _parse_type_from_string

if TYPE_CHECKING:
    from megg.rewrites.match_rewrites import Skeleton, SkeletonNode, SkeletonStmt

logger = logging.getLogger(__name__)

class ENodeDict(TypedDict):
    """Serialized e-node representation."""
    children: list[str]  # List of eclass IDs
    cost: float          # Cost of this node
    eclass: str          # Parent eclass ID
    op: str              # Operation name (e.g., "Add", "Lit", "Arg")
    subsumed: bool       # Whether this node is subsumed

class EClassDict(TypedDict):
    """Serialized e-class metadata."""
    type: str  # Type annotation

class EGraphDict(TypedDict):
    """Serialized e-graph representation."""
    nodes: dict[str, ENodeDict]       # Map from node ID to node data
    root_eclasses: list[str]          # List of root eclass IDs
    class_data: dict[str, EClassDict] # Map from eclass ID to eclass data

@dataclass
class MeggENode:
    """E-node in MeggEGraph."""
    node_id: str
    op: str
    children: List[str]  # List of eclass IDs
    eclass: str
    node_type: str  # "impl", "arg", "lit", or "unimplemented"
    value: Optional[Any] = None # for lit nodes, we need to store the value

@dataclass
class MeggEClass:
    """E-class in MeggEGraph."""
    eclass_id: str
    nodes: List[MeggENode]
    dtype: Optional[DataType] = None

@dataclass
class ExpressionNode:
    """Expression tree node extracted from e-graph."""
    op: str
    node_id: str
    eclass_id: str
    children: List[ExpressionNode]
    cost: float
    dtype: Optional[DataType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for MLIR conversion."""
        return {
            'op': self.op,
            'node_id': self.node_id,
            'eclass_id': self.eclass_id,
            'children': [c.to_dict() for c in self.children],
            'cost': self.cost,
            'type': self.dtype,
            'metadata': self.metadata
        }

@dataclass
class MeggEGraph:
    """Python-based E-graph for greedy extraction and custom instruction insertion."""

    enodes: Dict[str, MeggENode]
    eclasses: Dict[str, MeggEClass]
    root_eclasses: List[str]

    term_id_to_eclass: Dict[int, str] = field(default_factory=dict)

    cf_index: Dict[str, List[MeggENode]] = field(default_factory=dict)

    CF_OPS = [
        "Term.for_with_carry",
        "Term.for_",
        "Term.if_",
        "Term.while_",
        "Term.block"
    ]

    def _set_eclass_dtype(self, eclass_name: str, dtype: DataType):
        eclass = self.eclasses[eclass_name]
        if eclass.dtype is not None:
            assert eclass.dtype == dtype
        else:
            eclass.dtype = dtype

    def add_custom_instr_node(self, eclass_id: str, instr_name: str, operands: List[str] = None, result_type: str = "void") -> str:
        """在指定的 eclass 中添加一个 custom_instr node"""
        if eclass_id not in self.eclasses:
            raise ValueError(f"EClass {eclass_id} not found")

        operands = operands or []

        vec_node_idx = len(self.enodes)
        while True:
            vec_node_id = f"custom-vec-{vec_node_idx}"
            if vec_node_id not in self.enodes:
                break
            vec_node_idx += 1

        vec_eclass_idx = len(self.eclasses)
        while True:
            vec_eclass_id = f"Vec_Term-custom-{vec_eclass_idx}"
            if vec_eclass_id not in self.eclasses:
                break
            vec_eclass_idx += 1

        vec_node = MeggENode(
            node_id=vec_node_id,
            op="Vec",
            children=list(operands),
            eclass=vec_eclass_id,
            node_type="impl"
        )
        self.enodes[vec_node_id] = vec_node
        self.eclasses[vec_eclass_id] = MeggEClass(
            eclass_id=vec_eclass_id,
            nodes=[vec_node],
            dtype=None
        )

        custom_idx = len(self.enodes)
        while True:
            node_id = f"custom-instr-{instr_name}-{custom_idx}"
            if node_id not in self.enodes:
                break
            custom_idx += 1

        dtype = _parse_type_from_string(result_type)
        if dtype is None:
            raise ValueError(f"Unsupported instruction result type: {result_type}")

        custom_node = MeggENode(
            node_id=node_id,
            op="Term.custom_instr",
            children=[vec_eclass_id],
            eclass=eclass_id,
            node_type="impl",
            value=instr_name
        )

        self.enodes[node_id] = custom_node

        eclass = self.eclasses[eclass_id]
        eclass.nodes.append(custom_node)

        if eclass.dtype is None:
            eclass.dtype = dtype

        logger.info(
            f"Added custom_instr node '{instr_name}' to eclass {eclass_id} with {len(operands)} operands and type {result_type}"
        )
        return node_id

    def _build_control_flow_index(self):
        """Build control flow node index for fast skeleton matching."""
        self.cf_index = defaultdict(list)

        cf_count = 0
        for enode in self.enodes.values():
            if enode.op in self.CF_OPS:
                self.cf_index[enode.op].append(enode)
                cf_count += 1

        logger.info(f"Built control flow index: {cf_count} nodes across {len(self.cf_index)} types")
        for op_type, nodes in self.cf_index.items():
            logger.debug(f"  {op_type}: {len(nodes)} nodes")

    @classmethod
    def from_dict(cls, egraph_dict: EGraphDict) -> "MeggEGraph":
        enodes: Dict[str, MeggENode] = {}
        eclasses: Dict[str, MeggEClass] = {}
        unimplemented_enodes: Dict[str, List[str]] = {}

        @staticmethod
        def get_boxed_inner(node_name: str, egraph_dict: EGraphDict) -> str:
            name = egraph_dict["nodes"][node_name]["op"].strip('"')
            if name.startswith("Boxed(") and name.endswith(")"):
                name = name[len("Boxed("):-1].strip('"')
            return name

        @staticmethod
        def get_args(node_name: str, egraph_dict: EGraphDict) -> List[str]:
            vec_node = egraph_dict["nodes"][node_name]
            if vec_node["op"] != "Vec":
                return []
            return [egraph_dict["nodes"][c]["eclass"] for c in vec_node["children"]]

        @staticmethod
        def get_node_dtype(node_name: str, node_json: ENodeDict, egraph_dict: EGraphDict) -> Optional[DataType]:
            if not node_json["children"]:
                return None
            last_child = node_json["children"][-1]
            val = get_boxed_inner(last_child, egraph_dict)
            return _parse_type_from_string(val) if val else None
        
        @staticmethod
        def _convert_children_to_eclasses(children: List[str], egraph_dict: EGraphDict) -> List[str]:
            """Convert enode children to their eclass IDs."""
            return [egraph_dict["nodes"][child]["eclass"] for child in children]

        for class_name, class_json in egraph_dict['class_data'].items():
            class_type = class_json.get('type', 'unknown')
            if not ('Term' in class_type or 'Vec' in class_type):
                continue
            eclasses[class_name] = MeggEClass(eclass_id=class_name, nodes=[], dtype=None)

        for node_id, node_data in egraph_dict['nodes'].items():
            eclass = node_data['eclass']
            op = node_data['op']
            children = node_data['children']
            node_type = 'unimplemented'
            value = None
            dtype = None

            if op.startswith("Term."):
                op_stripped = op[len("Term."):]

                if op_stripped in ['arg', 'loop_index']:
                    value = get_boxed_inner(children[0], egraph_dict) if children else None
                    children = []
                    node_type = 'arg'
                elif op_stripped in ['loop_iter_arg', 'block_arg']:
                    _id = get_boxed_inner(children[0], egraph_dict) if children else None
                    idx = get_boxed_inner(children[1], egraph_dict) if len(children) > 1 else None
                    value = (_id, idx)
                    children = []
                    node_type = 'arg'
                elif op_stripped == 'lit':
                    if children:
                        lit_node_id = children[0]
                        lit_children = egraph_dict["nodes"][lit_node_id]["children"]
                        if lit_children:
                            value = get_boxed_inner(lit_children[0], egraph_dict)
                    children = []
                    node_type = 'lit'
                else:
                    children = [egraph_dict["nodes"][c]["eclass"] for c in children]
                    node_type = 'impl'
                    if op_stripped == 'cmp':
                        predcate_node = node_data['children'][2] if len(node_data['children']) > 2 else None
                        predicate = get_boxed_inner(predcate_node, egraph_dict) if predcate_node else None
                        if predicate and ':' in predicate:
                            predicate = predicate.split(':')[0].strip()
                        value = predicate
                    elif op_stripped == 'while_':
                        cond_block_node = node_data['children'][1] if len(node_data['children']) > 0 else None
                        body_block_node = node_data['children'][2] if len(node_data['children']) > 1 else None
                        cond_child = egraph_dict["nodes"][cond_block_node]["children"][0] if cond_block_node and cond_block_node in egraph_dict["nodes"] else None
                        body_child = egraph_dict["nodes"][body_block_node]["children"][0] if body_block_node and body_block_node in egraph_dict["nodes"] else None
                        cond_block_id = get_boxed_inner(cond_child, egraph_dict) if cond_child else None    
                        body_block_id = get_boxed_inner(body_child, egraph_dict) if body_block_node else None
                        value = (cond_block_id, body_block_id)
                    elif op_stripped == 'affine_for':
                        lower_map_node = node_data['children'][4] if len(node_data['children']) > 4 else None
                        upper_map_node = node_data['children'][5] if len(node_data['children']) > 5 else None
                        lower_map = get_boxed_inner(lower_map_node, egraph_dict) if lower_map_node else None
                        upper_map = get_boxed_inner(upper_map_node, egraph_dict) if upper_map_node else None
                        value = {
                            'lower_map': lower_map,
                            'upper_map': upper_map,
                            'has_iter_args': False,
                        }
                    elif op_stripped == 'affine_for_with_carry':
                        lower_map_node = node_data['children'][4] if len(node_data['children']) > 4 else None
                        upper_map_node = node_data['children'][5] if len(node_data['children']) > 5 else None
                        lower_map = get_boxed_inner(lower_map_node, egraph_dict) if lower_map_node else None
                        upper_map = get_boxed_inner(upper_map_node, egraph_dict) if upper_map_node else None
                        value = {
                            'lower_map': lower_map,
                            'upper_map': upper_map,
                            'has_iter_args': True,
                        }
                    elif op_stripped == 'custom_instr':
                        name_node = node_data['children'][0] if len(node_data['children']) > 0 else None
                        instr_name = get_boxed_inner(name_node, egraph_dict) if name_node else None
                        value = instr_name
                    elif op_stripped == 'component_instr':
                        name_node = node_data['children'][0] if len(node_data['children']) > 0 else None
                        comp_name = get_boxed_inner(name_node, egraph_dict) if name_node else None
                        value = comp_name
                    elif op_stripped == 'get_global':
                        global_name_node = node_data['children'][0] if len(node_data['children']) > 0 else None
                        value = get_boxed_inner(global_name_node, egraph_dict) if global_name_node else None
                    elif op_stripped == 'block':
                        value = get_boxed_inner(node_data['children'][0], egraph_dict)
                
                dtype = get_node_dtype(node_id, node_data, egraph_dict)
                if dtype and eclass in eclasses:
                    eclasses[eclass].dtype = dtype
            elif op == "Vec":
                node_type = 'impl'
                children=_convert_children_to_eclasses(node_data['children'], egraph_dict)
                
            else:
                if eclass not in unimplemented_enodes:
                    unimplemented_enodes[eclass] = []
                unimplemented_enodes[eclass].append(node_id)
            enodes[node_id] = MeggENode(
                node_id=node_id,
                op=op,
                children=children,
                eclass=eclass,
                node_type=node_type,
                value=value
            )
            if eclass in eclasses and node_type != 'unimplemented':
                eclasses[eclass].nodes.append(enodes[node_id])

        root_eclasses = [eclass_id for eclass_id in egraph_dict.get('root_eclasses', []) ]

        instance = cls(enodes=enodes, eclasses=eclasses, root_eclasses=root_eclasses)
        instance.unimplemented_enodes = unimplemented_enodes

        instance._build_control_flow_index()

        return instance

    @classmethod
    def from_egraph(cls, egraph, n_inline_leaves: int = 0,
                    split_primitive_outputs: bool = False,
                    func_transformer = None) -> MeggEGraph:
        """Construct MeggEGraph from egglog EGraph instance."""
        serialized = egraph._serialize(
            n_inline_leaves=n_inline_leaves,
            split_primitive_outputs=split_primitive_outputs
        )
        egraph_json_str = serialized.to_json()
        graph_dict: EGraphDict = json.loads(egraph_json_str)
        graph = cls.from_dict(graph_dict)
        for node in graph_dict['nodes'].values():
            if node['op'] == 'id_of':
                numeric_id = int(node['eclass'].removeprefix('i64-'))
                term_node = graph_dict['nodes'][node['children'][0]]
                graph.term_id_to_eclass[numeric_id] = term_node['eclass']
        if func_transformer is not None:
            block_id = func_transformer.get_id_of_term(func_transformer.top_block)
            root = graph.term_id_to_eclass.get(block_id)
            if root is None or root not in graph.eclasses:
                raise ValueError("Cannot resolve the function's top-level block root")
            graph.root_eclasses = [root]
        return graph

    def dump_to_json(self, filepath: str) -> None:
        """Dump MeggEGraph to JSON file for debugging."""
        data = {
            'enodes': {
                node_id: {
                    'op': enode.op,
                    'children': enode.children,
                    'eclass': enode.eclass,
                    'node_type': enode.node_type
                } for node_id, enode in self.enodes.items()
            },
            'eclasses': {
                eclass_id: {
                    'dtype': str(eclass.dtype) if eclass.dtype else None,
                    'nodes': [node.op for node in eclass.nodes]
                } for eclass_id, eclass in self.eclasses.items()
            },
            'root_eclasses': self.root_eclasses
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the e-graph."""
        total_nodes = len(self.enodes)
        total_eclasses = len(self.eclasses)

        nodes_by_type = {}
        for node in self.enodes.values():
            nodes_by_type[node.node_type] = nodes_by_type.get(node.node_type, 0) + 1

        return {
            'total_nodes': total_nodes,
            'total_eclasses': total_eclasses,
            'nodes_by_type': nodes_by_type,
            'root_eclasses': len(self.root_eclasses)
        }

class SkeletonMatcher:
    """Optimized skeleton matcher using control flow index."""

    SKELETON_TYPE_TO_OP = {
        "scf.for": ["Term.for_", "Term.for_with_carry"],
        "scf.if": ["Term.if_"],
        "scf.while": ["Term.while_"],
        "func.body": ["Term.block"]
    }

    def __init__(self, egraph: MeggEGraph, cost_function=None):
        self.egraph = egraph
        self.cost_function = cost_function
        self._input_extractor = None

    def match_skeleton(self, skeleton: 'Skeleton') -> List[tuple]:
        """Match skeleton using indexed search."""
        matches = []
        self._input_extractor = None

        op_types = self.SKELETON_TYPE_TO_OP.get(skeleton.root.container_type)
        if op_types is None:
            logger.warning(f"Unknown skeleton type: {skeleton.root.container_type}")
            return matches

        candidates = []
        for op_type in op_types:
            candidates.extend(self.egraph.cf_index.get(op_type, []))

        logger.info(f"Skeleton '{skeleton.instr_name}': "
                   f"found {len(candidates)} candidates for {skeleton.root.container_type}")
        logger.debug(f"  Op type queried: {op_type}")
        logger.debug(f"  Expected components: {list(skeleton.leaf_patterns.keys())}")

        stmt_to_full_name: Dict[str, str] = {}
        for full_name in skeleton.leaf_patterns.keys():
            suffix = full_name[len(skeleton.instr_name) + 1:] if full_name.startswith(f"{skeleton.instr_name}_") else full_name
            stmt_to_full_name[suffix] = full_name

        required_components = {
            comp_name for comp_name, _, _ in skeleton.operand_constraints
        }

        for candidate_node in candidates:
            bindings: Dict[str, str] = {}
            if not self._verify_skeleton_node(candidate_node, skeleton.root, bindings, skeleton, stmt_to_full_name):
                logger.debug(f"  Skeleton verification failed for eclass {candidate_node.eclass}")
                continue

            logger.info(f"  Bindings after skeleton verification: {bindings}")
            logger.info(f"  stmt_to_full_name mapping: {stmt_to_full_name}")

            component_operands: Dict[str, List[str]] = {}
            for stmt_name, stmt_eclass in bindings.items():
                full_name = stmt_to_full_name.get(stmt_name)
                logger.info(f"    Mapping stmt '{stmt_name}' to full_name '{full_name}'")
                if full_name:
                    operands = self._extract_component_operands(stmt_eclass)
                    logger.info(f"      Extracted {len(operands)} operands: {operands}")
                    component_operands[full_name] = operands

            logger.info(f"  component_operands keys: {list(component_operands.keys())}")
            logger.info(f"  Expected components: {list(skeleton.leaf_patterns.keys())}")

            missing = required_components - set(component_operands.keys())
            if missing:
                logger.info(f"  Missing components after verification: {missing}")
                continue

            var_bindings = self._verify_operand_constraints(component_operands, skeleton)
            if var_bindings is None:
                logger.debug(f"  Operand constraints failed for eclass {candidate_node.eclass}")
                continue

            if not self._inputs_available_before_candidate(candidate_node, var_bindings.values()):
                logger.debug(f"Inputs depend on replaced regions at {candidate_node.eclass}")
                continue

            matches.append((candidate_node.eclass, var_bindings))
            logger.info(f"✓ Matched skeleton '{skeleton.instr_name}' at eclass {candidate_node.eclass}")

        logger.info(f"Skeleton '{skeleton.instr_name}': {len(matches)} total matches")
        return matches

    def _inputs_available_before_candidate(self, candidate: MeggENode,
                                            operands: Iterable[str]) -> bool:
        """Conservatively require matched inputs to dominate the offload."""
        loop_ids = set()
        block_ids = set()
        visited = set()

        def region(node):
            if node.node_id in visited:
                return
            visited.add(node.node_id)
            children = []
            if node.op in ('Term.for_', 'Term.for_with_carry'):
                for binder in self.egraph.eclasses[node.children[3]].nodes:
                    if binder.op == 'Term.loop_index':
                        loop_ids.add(str(binder.value))
                children = [node.children[5] if node.op == 'Term.for_with_carry' else node.children[4]]
            elif node.op in ('Term.if_', 'Term.while_'):
                children = node.children[1:3]
            elif node.op == 'Term.block':
                block_ids.add(str(node.value))
                children = [child for child in node.children
                            if any(n.op == 'Vec' for n in self.egraph.eclasses.get(
                                child, MeggEClass(child, [])).nodes)]
            elif node.op == 'Vec':
                children = node.children
            for child in children:
                eclass = self.egraph.eclasses.get(child)
                if eclass:
                    for child_node in eclass.nodes:
                        region(child_node)

        region(candidate)
        # Check the selected representation, not every equivalent alternative.
        # An unused alternative can mention a removed binder (e.g. i - i).
        # Avoid the graph/extractor import cycle.
        from megg.egraph.extract import Extractor, MeggCost
        if self._input_extractor is None:
            self._input_extractor = Extractor(self.egraph, self.cost_function or MeggCost())
        visited.clear()
        try:
            pending = [self._input_extractor.find_best(operand).expr for operand in operands]
        except (ValueError, RecursionError):
            return False
        while pending:
            expr = pending.pop()
            if expr.node_id in visited:
                continue
            visited.add(expr.node_id)
            node = self.egraph.enodes[expr.node_id]
            if node.op == 'Term.loop_index' and str(node.value) in loop_ids:
                return False
            if node.op == 'Term.loop_iter_arg' and str(node.value[0]) in loop_ids:
                return False
            if node.op == 'Term.block_arg' and str(node.value[0]) in block_ids:
                return False
            if node.op in ('Term.load', 'Term.alloc', 'Term.alloca', 'Term.call',
                           'Term.custom_instr', 'Term.component_instr') or node.op in self.egraph.CF_OPS:
                return False
            pending.extend(expr.children)
        return True

    def _verify_operand_constraints(self, component_operands: Dict[str, List[str]],
                                    skeleton: 'Skeleton') -> Optional[Dict[str, str]]:
        """Verify operand equality constraints and return variable bindings."""
        if not skeleton.operand_constraints:
            return {}

        var_bindings: Dict[str, str] = {}

        for comp_name, operand_idx, var_name in skeleton.operand_constraints:
            operands = component_operands.get(comp_name)
            if operands is None or operand_idx >= len(operands):
                logger.debug(f"  ✗ Component '{comp_name}' missing operand index {operand_idx}")
                return None

            # Dependencies are not equivalent values: gain + 1 cannot become gain.
            operand_eclass = operands[operand_idx]
            if not self._validate_operand_type(var_name, operand_eclass, skeleton):
                logger.debug(
                    f"  ✗ Type check failed for variable '{var_name}' bound to eclass {operand_eclass}"
                )
                return None
            bound = var_bindings.get(var_name)
            if bound is None:
                var_bindings[var_name] = operand_eclass
            elif bound != operand_eclass:
                logger.debug(f"  ✗ Constraint failed: variable '{var_name}' bound to {bound} != {operand_eclass}")
                return None

        return var_bindings

    def _validate_operand_type(self, var_name: str, operand_eclass: str,
                                skeleton: 'Skeleton') -> bool:
        """Ensure operand types match expected argument types when available."""
        if not skeleton.arg_var_to_index or var_name not in skeleton.arg_var_to_index:
            return True

        expected_type = skeleton.arg_types.get(var_name)
        if expected_type is None:
            return True

        operand_class = self.egraph.eclasses.get(operand_eclass)
        if operand_class is None:
            logger.debug(f"  ✗ Missing eclass '{operand_eclass}' while type checking '{var_name}'")
            return False

        actual_type = operand_class.dtype
        if actual_type is None:
            logger.debug(
                f"  Warning: operand eclass '{operand_eclass}' has unknown dtype; skipping strict check for '{var_name}'"
            )
            return True

        if actual_type != expected_type:
            logger.debug(
                "  ✗ Type mismatch for '%s': expected %s, got %s",
                var_name,
                expected_type,
                actual_type,
            )
            return False

        return True

    def _extract_component_operands(self, stmt_eclass: str) -> List[str]:
        """Extract operand eclasses from a component instruction statement."""
        eclass = self.egraph.eclasses.get(stmt_eclass)
        if eclass is None:
            return []

        for node in eclass.nodes:
            if node.op != "Term.component_instr":
                continue

            best_children: Optional[List[str]] = None
            best_score = -1

            for child in node.children:
                if child not in self.egraph.eclasses:
                    continue

                vec_eclass = self.egraph.eclasses[child]
                for vec_node in vec_eclass.nodes:
                    if vec_node.op != "Vec":
                        continue

                    candidates = list(vec_node.children)
                    if not candidates:
                        continue

                    score = 0
                    for candidate in candidates:
                        ec = self.egraph.eclasses.get(candidate)
                        if not ec:
                            continue
                        if any(child_node.op == "Term.arg" for child_node in ec.nodes):
                            score += 1

                    if score > best_score:
                        best_children = candidates
                        best_score = score

            if best_children is not None:
                return best_children

        return []

    def _extract_body_eclass(self, control_node: MeggENode, container_type: str) -> Optional[str]:
        """Extract body block eclass from control flow node."""
        if container_type == "func.body":
            return control_node.eclass
        elif container_type == "scf.for":
            if len(control_node.children) >= 6:
                return control_node.children[5] if len(control_node.children) == 7 else control_node.children[4]
        elif container_type == "scf.if":
            if len(control_node.children) >= 2:
                return control_node.children[1]  # then_branch
        elif container_type == "scf.while":
            if len(control_node.children) >= 2:
                return control_node.children[1]  # before_block

        return None

    def _find_component_name(self, stmt_eclass: str) -> Optional[str]:
        """Extract component_instr name from statement eclass."""
        eclass = self.egraph.eclasses.get(stmt_eclass)
        if eclass is None:
            return None

        for node in eclass.nodes:
            if node.op == "Term.component_instr":
                return node.value

        return None

    def _is_control_flow_match(self, node: MeggENode, skeleton_node) -> bool:
        """Check whether an e-graph node matches the expected skeleton container type."""
        expected_ops = self.SKELETON_TYPE_TO_OP.get(skeleton_node.container_type)
        if expected_ops is None:
            return False
        return node.op in expected_ops

    def _verify_loop_bounds(self, code_node: 'MeggENode', expected_bounds: Dict[str, int]) -> bool:
        """验证循环边界是否匹配 (Bug Fix #2)"""
        if len(code_node.children) < 3:
            logger.debug(f"    Loop node has insufficient children: {len(code_node.children)}")
            return False

        lower_eclass = code_node.children[0]
        upper_eclass = code_node.children[1]
        step_eclass = code_node.children[2]

        actual_lower = self._extract_constant_from_eclass(lower_eclass)
        actual_upper = self._extract_constant_from_eclass(upper_eclass)
        actual_step = self._extract_constant_from_eclass(step_eclass)

        if actual_lower != expected_bounds.get('lower'):
            logger.debug(f"    Lower bound mismatch: {actual_lower} != {expected_bounds['lower']}")
            return False

        if actual_upper != expected_bounds.get('upper'):
            logger.debug(f"    Upper bound mismatch: {actual_upper} != {expected_bounds['upper']}")
            return False

        if actual_step != expected_bounds.get('step'):
            logger.debug(f"    Step mismatch: {actual_step} != {expected_bounds['step']}")
            return False

        logger.debug(f"    ✓ Loop bounds matched: lower={actual_lower}, upper={actual_upper}, step={actual_step}")
        return True

    def _extract_constant_from_eclass(self, eclass_id: str) -> Optional[int]:
        """从 eclass 中提取常量整数值 (Bug Fix #2)"""
        eclass = self.egraph.eclasses.get(eclass_id)
        if eclass is None:
            logger.debug(f"    Eclass {eclass_id} not found")
            return None

        for node in eclass.nodes:
            if node.op == "Term.lit":
                if node.value is not None:
                    try:
                        value_str = str(node.value)
                        match = re.search(r'(\d+)', value_str)
                        if match:
                            return int(match.group(1))
                        return int(node.value)
                    except (ValueError, TypeError) as e:
                        logger.debug(f"    Failed to parse constant from value '{node.value}': {e}")

                if len(node.children) > 0:
                    try:
                        lit_term_eclass = node.children[0]
                        lit_eclass = self.egraph.eclasses.get(lit_term_eclass)
                        if lit_eclass:
                            for lit_node in lit_eclass.nodes:
                                if 'int' in lit_node.op.lower():
                                    if lit_node.value is not None:
                                        return int(lit_node.value)
                    except Exception as e:
                        logger.debug(f"    Failed to extract from children: {e}")

        logger.debug(f"    No constant found in eclass {eclass_id}")
        return None

    def _verify_condition_match(self, code_eclass: str, pattern_term) -> bool:
        """Verify that code's condition eclass matches the pattern term."""
        pattern_text = str(pattern_term)
        cmp_str = self._extract_cmp_pattern_string(pattern_text)
        pattern_str = cmp_str if cmp_str is not None else pattern_text
        logger.debug(f"    Verifying condition: pattern={pattern_str}, code_eclass={code_eclass}")

        eclass = self.egraph.eclasses.get(code_eclass)
        if eclass is None:
            logger.debug(f"    ✗ Code eclass {code_eclass} not found")
            return False

        for node in eclass.nodes:
            if self._node_matches_pattern(node, pattern_term, pattern_str):
                logger.debug(f"    ✓ Found matching node: {node.op}")
                return True

        logger.debug(f"    ✗ No matching node found in eclass {code_eclass}")
        logger.debug(f"      Available nodes: {[n.op for n in eclass.nodes]}")
        return False

    def _node_matches_pattern(self, node: MeggENode, pattern_term, pattern_str: str) -> bool:
        """Check if an e-graph node matches a pattern term."""
        if not pattern_str.startswith("Term."):
            return False

        op_end = pattern_str.find("(")
        if op_end == -1:
            return False

        pattern_op = pattern_str[:op_end].strip()  # "Term.cmp"

        if node.op != pattern_op:
            return False

        if pattern_op == "Term.cmp":
            expected_pred = self._extract_cmp_predicate(pattern_str)
            if expected_pred is not None:
                actual_pred = node.value  # predicate存储在value中
                if actual_pred != expected_pred:
                    logger.debug(
                        "      ✗ Predicate mismatch: expected '%s', got '%s'",
                        expected_pred,
                        actual_pred,
                    )
                    return False
                logger.debug(f"      ✓ Predicate matched: '{expected_pred}'")

        return True

    def _extract_cmp_pattern_string(self, term_str: str) -> Optional[str]:
        """Return canonical `Term.cmp(...)` portion from a pattern string."""
        idx = term_str.find("Term.cmp(")
        if idx == -1:
            return None

        start = idx
        depth = 0
        in_string = False
        escape = False

        for i in range(idx, len(term_str)):
            ch = term_str[i]

            if escape:
                escape = False
                continue

            if ch == '\\' and in_string:
                escape = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if not in_string:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        return term_str[start:i + 1]

        return term_str[start:]

    def _extract_cmp_predicate(self, pattern_str: str) -> Optional[str]:
        """Safely extract the predicate (3rd argument) from a Term.cmp pattern string."""
        if not pattern_str.startswith("Term.cmp"):
            return None

        start = pattern_str.find('(')
        end = pattern_str.rfind(')')
        if start == -1 or end == -1 or end <= start:
            return None

        args_str = pattern_str[start + 1:end]
        args: List[str] = []
        current: List[str] = []
        depth = 0
        in_string = False
        escape = False

        for ch in args_str:
            if escape:
                current.append(ch)
                escape = False
                continue

            if ch == '\\' and in_string:
                current.append(ch)
                escape = True
                continue

            if ch == '"':
                in_string = not in_string
                current.append(ch)
                continue

            if not in_string:
                if ch in '([{':
                    depth += 1
                elif ch in ')]}':
                    if depth > 0:
                        depth -= 1
                elif ch == ',' and depth == 0:
                    args.append(''.join(current).strip())
                    current = []
                    continue

            current.append(ch)

        if current:
            args.append(''.join(current).strip())

        if len(args) >= 3:
            predicate_expr = args[2].strip()
            if predicate_expr.startswith('"') and predicate_expr.endswith('"'):
                return predicate_expr.strip('"')
            match = re.match(r'"([^\"]+)"', predicate_expr)
            if match:
                return match.group(1)

        logger.debug(
            "      ⚠ Unable to parse predicate from pattern '%s' (args=%s)",
            pattern_str,
            args,
        )
        return None

    def _verify_skeleton_node(self,
                              node: MeggENode,
                              skeleton_node,
                              bindings: Dict[str, str],
                              skeleton: 'Skeleton',
                              stmt_to_full_name: Dict[str, str]) -> bool:
        """Verify skeleton node structure and control flow parameters."""
        if not self._is_control_flow_match(node, skeleton_node):
            logger.info(f"  Control flow type mismatch: {node.op} vs {skeleton_node.container_type}")
            return False

        logger.info(f"  ✓ Matched control flow type: {skeleton_node.container_type} (op={node.op})")

        if skeleton_node.container_type == "scf.for" and skeleton_node.loop_bounds:
            if not self._verify_loop_bounds(node, skeleton_node.loop_bounds):
                logger.info(f"  ✗ Loop bounds mismatch")
                logger.info(f"     Expected: {skeleton_node.loop_bounds}")
                return False
            logger.info(f"  ✓ Loop bounds matched: {skeleton_node.loop_bounds}")

        if skeleton_node.condition_term is not None:
            if skeleton_node.container_type == "scf.if":
                if len(node.children) < 1:
                    logger.info(f"  ✗ scf.if node missing condition (no children)")
                    return False
                code_condition_eclass = node.children[0]
                if not self._verify_condition_match(code_condition_eclass, skeleton_node.condition_term):
                    logger.info(f"  ✗ scf.if condition mismatch")
                    logger.info(f"     Expected pattern: {skeleton_node.condition_term}")
                    logger.info(f"     Code eclass: {code_condition_eclass}")
                    return False
                logger.info(f"  ✓ scf.if condition matched: {skeleton_node.condition_term}")

        logger.info(f"  Verifying skeleton node {skeleton_node.container_type} with {len(skeleton_node.blocks)} blocks")

        for skeleton_block in skeleton_node.blocks:
            block_eclass_id = self._find_block_for_skeleton(node, skeleton_node.container_type, skeleton_block.name)
            if block_eclass_id is None:
                logger.info(f"  ✗ Block '{skeleton_block.name}' not found in node {node.op}")
                return False

            logger.info(f"  ✓ Found block '{skeleton_block.name}' at eclass {block_eclass_id}")

            block_stmts = self._get_block_statements(block_eclass_id)
            if block_stmts is None:
                logger.info(f"  ✗ No statements in block '{skeleton_block.name}'")
                return False

            logger.info(f"  Block '{skeleton_block.name}' has {len(block_stmts)} statements, expecting {len(skeleton_block.statements)}")

            if not self._verify_block_statements(block_stmts,
                                                 skeleton_block.statements,
                                                 bindings,
                                                 skeleton,
                                                 stmt_to_full_name):
                logger.info(f"  ✗ Statement verification failed for block '{skeleton_block.name}'")
                return False

        logger.debug(f"Successfully verified skeleton node {skeleton_node.container_type}")
        return True

    def _verify_block_statements(self,
                                block_stmts: List[str],
                                skeleton_stmts,
                                bindings: Dict[str, str],
                                skeleton: 'Skeleton',
                                stmt_to_full_name: Dict[str, str]) -> bool:
        """验证 block 的 statements 是否匹配 skeleton"""
        # TODO: 可以添加顺序验证

        logger.info(f"    Verifying {len(skeleton_stmts)} skeleton statements against {len(block_stmts)} block statements")

        last_matched_index = -1

        for skeleton_stmt in skeleton_stmts:
            if skeleton_stmt.is_leaf():
                logger.info(f"    Looking for leaf statement '{skeleton_stmt.name}'")

                full_name = stmt_to_full_name.get(skeleton_stmt.name)
                if full_name is None and skeleton.instr_name:
                    full_name = f"{skeleton.instr_name}_{skeleton_stmt.name}"

                has_constraints = any(
                    comp_name == full_name for comp_name, _, _ in skeleton.operand_constraints
                )

                if not has_constraints:
                    logger.info(f"      ⊘ Skipping leaf '{skeleton_stmt.name}' (no constraints, component may be empty-operand)")
                    continue

                found = False
                for idx in range(last_matched_index + 1, len(block_stmts)):
                    stmt_eclass = block_stmts[idx]
                    if self._is_component_match_by_stmt(stmt_eclass, skeleton_stmt):
                        bindings[skeleton_stmt.name] = stmt_eclass
                        logger.info(f"      ✓ Found leaf '{skeleton_stmt.name}' at {stmt_eclass}")
                        last_matched_index = idx
                        found = True
                        break

                if not found:
                    logger.info(f"      ✗ Leaf statement '{skeleton_stmt.name}' not found")
                    return False

            elif skeleton_stmt.is_nested():
                logger.info(f"    Looking for nested statement '{skeleton_stmt.name}' (type={skeleton_stmt.nested_skeleton.container_type})")
                found = False
                for idx in range(last_matched_index + 1, len(block_stmts)):
                    stmt_eclass = block_stmts[idx]
                    eclass = self.egraph.eclasses.get(stmt_eclass)
                    if eclass:
                        for nested_node in eclass.nodes:
                            if self._verify_skeleton_node(nested_node, skeleton_stmt.nested_skeleton, bindings, skeleton, stmt_to_full_name):
                                bindings[skeleton_stmt.name] = stmt_eclass
                                logger.info(f"      ✓ Found nested '{skeleton_stmt.name}' at {stmt_eclass}")
                                last_matched_index = idx
                                found = True
                                break
                    if found:
                        break

                if not found:
                    logger.debug(f"Nested control flow '{skeleton_stmt.name}' not found")
                    return False

        return True

    def _is_component_match_by_stmt(self, stmt_eclass_id: str, skeleton_stmt) -> bool:
        """检查 statement eclass 是否匹配 skeleton_stmt（通过 component_instr）"""
        eclass = self.egraph.eclasses.get(stmt_eclass_id)
        if eclass is None:
            return False

        for node in eclass.nodes:
            if node.op == "Term.component_instr":
                if len(node.children) > 0:
                    name_eclass_id = node.children[0]
                    logger.info(f"        Found component_instr with name eclass: {name_eclass_id}")
                    return True

        return False

    def _find_block_for_skeleton(self, node: MeggENode, container_type: str, block_name: str) -> Optional[str]:
        """根据 skeleton block name 找到对应的 MLIR block eclass"""
        if container_type == "func.body" or container_type == "scf.for":
            if block_name == "body":
                return self._find_body_block(node, container_type)

        elif container_type == "scf.if":
            if block_name == "then" and len(node.children) >= 2:
                return node.children[1]
            elif block_name == "else" and len(node.children) >= 3:
                return node.children[2]

        elif container_type == "scf.while":
            if block_name == "before" and len(node.children) >= 2:
                return node.children[1]
            elif block_name == "after" and len(node.children) >= 3:
                return node.children[2]

        return None

    def _find_body_block(self, control_node: MeggENode, control_type: str) -> Optional[str]:
        """从控制流节点中找到body block的eclass"""
        if control_type == "func.body":
            return control_node.eclass
        elif control_type == "scf.for":
            if len(control_node.children) == 6:
                return control_node.children[4]  # For
            elif len(control_node.children) == 7:
                return control_node.children[5]  # ForWithCarry
        elif control_type == "scf.if":
            if len(control_node.children) >= 2:
                return control_node.children[1]  # then_block
        elif control_type == "scf.while":
            if len(control_node.children) >= 2:
                return control_node.children[1]  # before_block

        return None

    def _get_block_statements(self, block_eclass_id: str) -> Optional[List[str]]:
        """从block eclass中提取语句列表"""
        eclass = self.egraph.eclasses.get(block_eclass_id)
        if eclass is None:
            logger.info(f"      ✗ Block eclass '{block_eclass_id}' not found")
            return None

        for node in eclass.nodes:
            if node.op == "Term.block":
                if len(node.children) >= 2:
                    stmts_vec_eclass = node.children[1]
                    statements = self._extract_vec_elements(stmts_vec_eclass)
                    _yield = self.egraph.eclasses.get(statements[-1]).nodes[0]
                    if _yield.op == 'Term.yield_':
                        for op in _yield.children:
                            op = self.egraph.eclasses.get(op)
                            if op is None:
                                continue
                            assert op.nodes[0].op == 'Vec'
                            if len(op.nodes[0].children)==0:
                                statements=statements[:-1]
                                break
                    logger.info(f"      Found {len(statements)} statements in Vec at {stmts_vec_eclass}")
                    return statements

        logger.info(f"      ✗ No Term.block node in eclass '{block_eclass_id}'")
        return None

    def _extract_vec_elements(self, vec_eclass_id: str) -> List[str]:
        """从Vec eclass中提取所有元素的eclass列表"""
        eclass = self.egraph.eclasses.get(vec_eclass_id)
        if eclass is None:
            return []

        for node in eclass.nodes:
            if node.op == "Vec":
                return node.children

        return []

    def _is_component_match(self, stmt_eclass_id: str, instr_name: str, comp_name: str) -> bool:
        """检查语句eclass是否匹配组件名称"""
        eclass = self.egraph.eclasses.get(stmt_eclass_id)
        if eclass is None:
            return False

        expected_name = f"{instr_name}_{comp_name}"

        for node in eclass.nodes:
            if node.op == "ComponentInstr":
                if hasattr(node, 'value') and node.value:
                    node_name = str(node.value).strip('"')
                    if node_name == expected_name:
                        return True

        return False

    def _verify_order(self, component_positions: Dict[str, int],
                     order_constraints: List[tuple]) -> bool:
        """验证组件在block中的顺序是否满足约束"""
        for comp_a, comp_b in order_constraints:
            if comp_a not in component_positions or comp_b not in component_positions:
                return False
            if component_positions[comp_a] >= component_positions[comp_b]:
                return False  # 违反顺序约束

        return True
