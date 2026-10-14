"""Greedy Extraction from E-graphs with Customizable Cost Functions."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, TypeVar

from megg.egraph.megg_egraph import ExpressionNode, MeggEClass, MeggEGraph, MeggENode

Cost = TypeVar('Cost')  # Cost can be any comparable type (float, int, etc.)

class CostFunction(ABC, Generic[Cost]):
    """Abstract base class for cost functions."""

    @abstractmethod
    def cost(self, node: MeggENode, costs: Callable[[str], Cost]) -> Cost:
        """Calculate the cost of an e-node."""
        pass

    def cost_rec(self, expr: ExpressionNode) -> Cost:
        """Calculate the total cost of an expression tree."""
        cost_map: Dict[str, Cost] = {}
        
        def compute_cost(node: ExpressionNode) -> Cost:
            if node.eclass_id in cost_map:
                return cost_map[node.eclass_id]
                
            child_costs = [compute_cost(child) for child in node.children]
            
            mock_node = MeggENode(
                node_id=node.node_id,
                op=node.op,
                children=[child.eclass_id for child in node.children],
                eclass=node.eclass_id,
                node_type=node.metadata.get('node_type', 'expr')
            )
            
            def cost_lookup(eclass_id: str) -> Cost:
                for child in node.children:
                    if child.eclass_id == eclass_id:
                        return compute_cost(child)
                raise ValueError(f"Child eclass {eclass_id} not found")
            
            node_cost = self.cost(mock_node, cost_lookup)
            cost_map[node.eclass_id] = node_cost
            return node_cost
        
        return compute_cost(expr)

class AstSize(CostFunction[int]):
    """Simple cost function that counts total AST size."""

    def cost(self, node: MeggENode, costs: Callable[[str], int]) -> int:
        total = 1
        for child_eclass in node.children:
            total += costs(child_eclass)
        return total

class AstDepth(CostFunction[int]):
    """Simple cost function that counts maximum AST depth."""

    def cost(self, node: MeggENode, costs: Callable[[str], int]) -> int:
        max_child = 0
        for child_eclass in node.children:
            max_child = max(max_child, costs(child_eclass))
        return 1 + max_child

class OpWeightedCost(CostFunction[float]):
    """Cost function with custom weights per operation."""

    def __init__(self, op_costs: Dict[str, float], default_cost: float = 1.0):
        """Initialize with operation-specific costs."""
        self.op_costs = op_costs
        self.default_cost = default_cost

    def cost(self, node: MeggENode, costs: Callable[[str], float]) -> float:
        op_cost = self.op_costs.get(node.op, self.default_cost)

        total = op_cost
        for child_eclass in node.children:
            total += costs(child_eclass)

        return total

class ConstantFoldingCost(CostFunction[float]):
    """Cost function that strongly prefers constant-folded expressions."""

    def cost(self, node: MeggENode, costs: Callable[[str], float]) -> float:
        if node.op == 'Lit':
            return 1.0

        base_cost = 5.0
        total = base_cost
        for child_eclass in node.children:
            total += costs(child_eclass)

        return total

class MeggCost(CostFunction[float]):
    """Megg-specific cost function that assigns different costs to different operations."""

    OP_COSTS = {
        'LitTerm.int': 1.0,
        'LitTerm.float': 1.0,
        'Term.arg': 1.0,
        'Term.loop_index': 1.0,
        'Term.loop_iter_arg': 1.0,
        'Term.block_arg': 1.0,

        'Term.neg': 2.0,

        'Term.add': 2.0,
        'Term.sub': 2.0,

        'Term.mul': 4.0,
        'Term.div': 8.0,
        'Term.rem': 8.0,

        'Term.index_cast': 100.0,
        'Term.sitofp': 5.0,
        'Term.uitofp': 5.0,
        'Term.fptosi': 5.0,
        'Term.fptoui': 5.0,
        'Term.extsi': 3.0,
        'Term.extui': 3.0,
        'Term.trunci': 3.0,
        'Term.bitcast': 2.0,

        'Term.cmp': 3.0,

        'Term.if_': 10.0,
        'Term.while_': 20.0,
        'Term.for_': 15.0,
        'Term.for_with_carry': 18.0,
        'Term.affine_for': 15.0,
        'Term.affine_for_with_carry': 18.0,
        'Term.block': 5.0,
        'Term.yield_': 3.0,
        'Term.affine_yield': 3.0,
        'Term.condition': 5.0,

        'Term.alloc': 10.0,
        'Term.get_global': 5.0,
        'Term.store': 8.0,
        'Term.load': 6.0,

        'Term.return_': 2.0,

        'Term.custom_instr': 0.1,

        # Component instructions (INFINITE COST - should never be extracted!)
        'Term.component_instr': float('inf'),

        'Vec': 0.0,
    }

    def __init__(self, custom_costs: Optional[Dict[str, float]] = None):
        """Initialize MeggCost with optional custom operation costs."""
        self.op_costs = self.OP_COSTS.copy()
        if custom_costs:
            self.op_costs.update(custom_costs)

    def cost(self, node: MeggENode, costs: Callable[[str], float]) -> float:
        """Calculate the cost of a node based on its operation type and children."""
        op_cost = self.op_costs.get(node.op, 5.0)  # Default 5.0 for unknown ops

        total = op_cost
        for child_eclass in node.children:
            total += costs(child_eclass)

        return total

class AffineCost(CostFunction[float]):
    """Cost function that strongly prefers affine operations."""

    AFFINE_OPS = {
        'Term.add',
        'Term.sub',
        'Term.mul',
        'Term.neg',

        'LitTerm.int',
        'LitTerm.float',
        'Term.arg',
        'Term.loop_index',
        'Term.loop_iter_arg',
        'Term.block_arg',

        'Term.index_cast',
        'Term.extsi',
        'Term.extui',
        'Term.trunci',

        'Vec',
    }

    NON_AFFINE_COST = 10000.0

    def __init__(self, affine_cost: float = 0.0, non_affine_cost: float = 10000.0):
        """Initialize AffineCost with customizable costs."""
        self.affine_cost = affine_cost
        self.non_affine_cost = non_affine_cost

    def cost(self, node: MeggENode, costs: Callable[[str], float]) -> float:
        """Calculate the cost of a node based on whether it's affine or not."""
        if node.op in self.AFFINE_OPS:
            op_cost = self.affine_cost
        else:
            op_cost = self.non_affine_cost

        total = op_cost
        for child_eclass in node.children:
            total += costs(child_eclass)

        return total

@dataclass
class ExtractionResult(Generic[Cost]):
    """Result of extraction from an e-class."""
    cost: Cost
    expr: ExpressionNode
    eclass_id: str

class Extractor(Generic[Cost]):
    """Greedy extractor for e-graphs with customizable cost functions."""

    def __init__(self, egraph: MeggEGraph, cost_function: CostFunction[Cost]):
        """Create a new Extractor with the given cost function."""
        self.egraph = egraph
        self.cost_function = cost_function
        self.costs: Dict[str, (Cost, MeggENode)] = {}
        self.logger = logging.getLogger(__name__)
        self.find_costs()

    def find_best(self, eclass_id: Optional[str] = None) -> ExtractionResult[Cost]:
        """Find the cheapest (lowest cost) expression in the given e-class."""
        if eclass_id is None:
            if not self.egraph.root_eclasses:
                raise ValueError("No root eclasses available")
            eclass_id = self.egraph.root_eclasses[0]

        if eclass_id not in self.costs:
            raise ValueError(f"E-class {eclass_id} has no computed cost")

        cost, best_node = self.costs[eclass_id]

        expr = self.build_recexpr(eclass_id, best_node)

        return ExtractionResult(cost=cost, expr=expr, eclass_id=eclass_id)

    def find_best_node(self, eclass_id: str) -> MeggENode:
        """Find the cheapest e-node in the given e-class."""
        if eclass_id not in self.costs:
            raise ValueError(f"E-class {eclass_id} has no computed cost")

        _, best_node = self.costs[eclass_id]
        return best_node

    def find_best_cost(self, eclass_id: str) -> Cost:
        """Find the cost of the term that would be extracted from this e-class."""
        if eclass_id not in self.costs:
            raise ValueError(f"E-class {eclass_id} has no computed cost")

        cost, _ = self.costs[eclass_id]
        return cost

    def build_recexpr(self, eclass_id: str, node: MeggENode) -> ExpressionNode:
        """Build expression tree recursively from best nodes."""
        children_exprs = []
        for child_eclass in node.children:
            if child_eclass not in self.costs:
                raise ValueError(f"Child e-class {child_eclass} has no computed cost")
                
            _, child_best_node = self.costs[child_eclass]
            child_expr = self.build_recexpr(child_eclass, child_best_node)
            children_exprs.append(child_expr)

        cost_val = self.costs[eclass_id][0]
        cost_float = float(cost_val) if isinstance(cost_val, (int, float)) else 0.0
        
        eclass = self.egraph.eclasses[eclass_id]
        
        normalized_op = self.normalize_op_name(node.op)

        en = ExpressionNode(
            op=normalized_op,
            node_id=node.node_id,
            eclass_id=eclass_id,
            children=children_exprs,
            cost=cost_float,
            dtype=eclass.dtype,
            metadata={'node_type': node.node_type}
        )
        if node.node_type == 'lit':
            en.metadata['value'] = node.value
        elif node.node_type == 'arg':
            en.metadata['arg_info'] = node.value
        elif node.op == 'Term.cmp':
            en.metadata['cmp_info'] = node.value
        elif node.op == 'Term.while_':
            en.metadata['while_info'] = node.value
        elif node.op == 'Term.get_global':
            en.metadata['global_name'] = node.value
        elif node.op in ('Term.affine_for', 'Term.affine_for_with_carry'):
            en.metadata['affine_info'] = node.value or {}
        elif node.op == 'Term.custom_instr':
            en.metadata['instr_name'] = node.value
        
        return en

    def node_total_cost(self, node: MeggENode) -> Optional[Cost]:
        """Calculate total cost of a node if all children have costs."""

        def is_legal_node(op: str) -> bool:
            if op.startswith("String") or op.startswith("i64"):
                return False
            return True
        node.children = [child for child in node.children if is_legal_node(child)]
        for child_eclass in node.children:
            
            if child_eclass not in self.costs:
                return None
        
        def cost_lookup(eclass_id: str) -> Cost:
            cost, _ = self.costs[eclass_id]
            return cost
        
        return self.cost_function.cost(node, cost_lookup)

    def find_costs(self):
        """Iteratively compute costs for all e-classes."""
        did_something = True
        iterations = 0
        max_iterations = len(self.egraph.eclasses) * 10  # Safety limit

        while did_something and iterations < max_iterations:
            did_something = False
            iterations += 1

            for eclass_id, eclass in self.egraph.eclasses.items():
                new_cost_node = self.make_pass(eclass)

                if new_cost_node is None:
                    continue

                new_cost, new_node = new_cost_node

                if eclass_id not in self.costs:
                    self.costs[eclass_id] = (new_cost, new_node)
                    did_something = True
                else:
                    old_cost, _ = self.costs[eclass_id]
                    if self.compare_costs(new_cost, old_cost) < 0:
                        self.costs[eclass_id] = (new_cost, new_node)
                        did_something = True

        for eclass_id in self.egraph.eclasses:
            if eclass_id not in self.costs:
                self.logger.warning(
                    f"Failed to compute cost for eclass {eclass_id}: {self.egraph.eclasses[eclass_id].nodes}"
                )
            else:
                pass

        self.logger.info(f"Cost computation converged in {iterations} iterations")

    def make_pass(self, eclass: MeggEClass) -> Optional[tuple[Cost, MeggENode]]:
        """Try to find the best node in an e-class for this iteration."""
        best_cost = None
        best_node = None

        for node in eclass.nodes:

            node_cost = self.node_total_cost(node)
            if node_cost is None:
                continue

            if best_cost is None or self.compare_costs(node_cost, best_cost) < 0:
                best_cost = node_cost
                best_node = node

        if best_cost is not None and best_node is not None:
            return (best_cost, best_node)

        return None

    def normalize_op_name(self, op: str) -> str:
        """Normalize operation name from egglog format to MLIR format."""
        if op.startswith('Term.'):
            op = op[5:]  # Remove 'Term.'

        if op and op[0].islower():
            op = op[0].upper() + op[1:]

        return op

    def compare_costs(self, a: Cost, b: Cost) -> int:
        """Compare two costs."""
        return (a > b) - (a < b)
