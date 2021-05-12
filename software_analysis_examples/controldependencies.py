#
#  SPDX-FileCopyrightText: 2021 Chair of Software Engineering II, University of Passau
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
"""Provides the implementation of a control-dependence graph.

To compute control dependencies, one has to build a control-flow graph and a
post-dominator tree, which are also implemented in this module.
"""
from __future__ import annotations

import queue
import sys
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, TypeVar, cast

import networkx as nx  # type: ignore
from bytecode import BasicBlock, Bytecode, ControlFlowGraph  # type: ignore
from networkx.drawing.nx_pydot import to_pydot  # type: ignore


class ProgramGraphNode:
    """A base class for a node of the program graph."""

    def __init__(
        self,
        index: int,
        basic_block: Optional[BasicBlock] = None,
        is_artificial: bool = False,
    ) -> None:
        self._index = index
        self._basic_block = basic_block
        self._is_artificial = is_artificial
        self._predicate_id: Optional[int] = None

    @property
    def index(self) -> int:
        """Provides the index of the node.

        Returns:
            The index of the node
        """
        return self._index

    @property
    def basic_block(self) -> Optional[BasicBlock]:
        """Provides the basic block attached to this node.

        Returns:
            The optional basic block attached to this node
        """
        return self._basic_block

    @property
    def is_artificial(self) -> bool:
        """Whether or not a node is artificially inserted into the graph.

        Returns:
            Whether or not a node is artificially inserted into the graph
        """
        return self._is_artificial

    @property
    def predicate_id(self) -> Optional[int]:
        """If this node creates a branch based on a predicate, than this stores the id
        of this predicate.

        Returns:
            The predicate id assigned to this node, if any.
        """
        return self._predicate_id

    @predicate_id.setter
    def predicate_id(self, predicate_id: int) -> None:
        """Set a new predicate id.

        Args:
            predicate_id: The predicate id
        """
        self._predicate_id = predicate_id

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ProgramGraphNode):
            return False
        if self is other:
            return True
        return self._index == other.index

    def __hash__(self) -> int:
        return 31 + 17 * self._index

    def __str__(self) -> str:
        return f"ProgramGraphNode({self._index})"

    def __repr__(self) -> str:
        return f"ProgramGraphNode(index={self._index}, basic_block={self._basic_block})"


N = TypeVar("N", bound=ProgramGraphNode)  # pylint: disable=invalid-name


class ProgramGraph(Generic[N]):
    """Provides a base implementation for a program graph.

    Internally, this program graph uses the `NetworkX` library to hold the graph and
    do all the operations on it.
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()

    def add_node(self, node: N, **attr: Any) -> None:
        """Add a node to the graph

        Args:
            node: The node
            attr: A dict of attributes that will be attached to the node
        """
        self._graph.add_node(node, **attr)

    def add_edge(self, start: N, end: N, **attr: Any) -> None:
        """Add an edge between two nodes to the graph

        Args:
            start: The start node of the edge
            end: The end node of the edge
            attr: A dict of attributes that will be attached to the edge.
        """
        self._graph.add_edge(start, end, **attr)

    def get_predecessors(self, node: N) -> Set[N]:
        """Provides a set of all direct predecessors of a node.

        Args:
            node: The node to start

        Returns:
            A set of direct predecessors of the node
        """
        predecessors: Set[N] = set()
        for predecessor in self._graph.predecessors(node):
            predecessors.add(predecessor)
        return predecessors

    def get_successors(self, node: N) -> Set[N]:
        """Provides a set of all direct successors of a node.

        Args:
            node: The node to start

        Returns:
            A set of direct successors of the node
        """
        successors: Set[N] = set()
        for successor in self._graph.successors(node):
            successors.add(successor)
        return successors

    @property
    def nodes(self) -> Set[N]:
        """Provides all nodes in the graph.

        Returns:
            The set of all nodes in the graph
        """
        return {
            node
            for node in self._graph.nodes  # pylint: disable=unnecessary-comprehension
        }

    @property
    def graph(self) -> nx.DiGraph:
        """The internal graph.

        Returns:
            The internal graph
        """
        return self._graph

    @property
    def entry_node(self) -> Optional[N]:
        """Provides the entry node of the graph.

        Returns:
            The entry node of the graph
        """
        for node in self._graph.nodes:
            if len(self.get_predecessors(node)) == 0:
                return node
        return None

    @property
    def exit_nodes(self) -> Set[N]:
        """Provides the exit nodes of the graph.

        Returns:
            The set of exit nodes of the graph
        """
        exit_nodes: Set[N] = set()
        for node in self._graph.nodes:
            if len(self.get_successors(node)) == 0:
                exit_nodes.add(node)
        return exit_nodes

    def get_transitive_successors(self, node: N) -> Set[N]:
        """Calculates the transitive closure (the transitive successors) of a node.

        Args:
            node: The node to start with

        Returns:
            The transitive closure of the node
        """
        return self._get_transitive_successors(node, set())

    def _get_transitive_successors(self, node: N, done: Set[N]) -> Set[N]:
        successors: Set[N] = set()
        for successor_node in self.get_successors(node):
            if successor_node not in done:
                successors.add(successor_node)
                done.add(successor_node)
                successors.update(self._get_transitive_successors(successor_node, done))
        return successors

    def get_least_common_ancestor(self, first: N, second: N) -> N:
        """Calculates the least or lowest common ancestor node of two nodes of the
        graph.

        Both nodes have to be part of the graph!

        Args:
            first: The first node
            second: The second node

        Returns:
            The least common ancestor node of the two nodes
        """
        return nx.lowest_common_ancestor(self._graph, first, second)

    @property
    def dot(self) -> str:
        """Provides the DOT representation of this graph.

        Returns:
            The DOT representation of this graph
        """
        dot = to_pydot(self._graph)
        return dot.to_string()


G = TypeVar("G", bound=ProgramGraph)  # pylint: disable=invalid-name


def filter_dead_code_nodes(graph: G, entry_node_index: int = 0) -> G:
    """Prunes dead nodes from the given graph.

    A dead node is a node that has no entry node.  To specify a legal entry node,
    one can use the `entry_node_index` parameter.

    Args:
        graph: The graph to prune nodes from
        entry_node_index: The index of the valid entry node

    Returns:
        The graph without the pruned dead nodes
    """
    has_changed = True
    while has_changed:
        # Do this until we have reached a fixed point, i.e., removed all dead
        # nodes from the graph.
        has_changed = False
        for node in graph.nodes:
            if graph.get_predecessors(node) == set() and node.index != entry_node_index:
                # The only node in the graph that is allowed to have no predecessor
                # is the entry node, i.e., the node with index 0.  All other nodes
                # without predecessors are considered dead code and thus removed.
                graph.graph.remove_node(node)
                has_changed = True
    return graph


class CFG(ProgramGraph[ProgramGraphNode]):
    """The control-flow graph implementation based on the program graph."""

    # Attribute where the predicate id of the instrumentation is stored
    PREDICATE_ID: str = "predicate_id"

    def __init__(self, bytecode_cfg: ControlFlowGraph):
        """Create new CFG. Do not call directly, use static factory methods.

        Args:
            bytecode_cfg: the control flow graph of the underlying bytecode.
        """
        super().__init__()
        self._bytecode_cfg = bytecode_cfg
        self._diameter: Optional[int] = None

    @staticmethod
    def from_bytecode(bytecode: Bytecode) -> CFG:
        """Generates a new control-flow graph from a bytecode segment.

        Besides generating a node for each block in the bytecode segment, as returned by
        `bytecode`'s `ControlFlowGraph` implementation, we add two artificial nodes to
        the generated CFG:
         - an artificial entry node, having index -1, that is guaranteed to fulfill the
           property of an entry node, i.e., there is no incoming edge, and
         - an artificial exit node, having index `sys.maxsize`, that is guaranteed to
           fulfill the property of an exit node, i.e., there is no outgoing edge, and
           that is the only such node in the graph, which is important, e.g., for graph
           reversal.
        The index values are chosen that they do not appear in regular graphs, thus one
        can easily distinguish them from the normal nodes in the graph by checking for
        their index-property's value.

        Args:
            bytecode: The bytecode segment

        Returns:
            The control-flow graph for the segment
        """
        blocks = ControlFlowGraph.from_bytecode(bytecode)
        cfg = CFG(blocks)

        # Create the nodes and a mapping of all edges to generate
        edges, nodes = CFG._create_nodes(blocks)

        # Insert all edges between the previously generated nodes
        CFG._create_graph(cfg, edges, nodes)

        # Filter all dead-code nodes
        cfg = filter_dead_code_nodes(cfg)

        # Insert dummy exit and entry nodes
        cfg = CFG._insert_dummy_exit_node(cfg)
        cfg = CFG._insert_dummy_entry_node(cfg)
        return cfg

    def bytecode_cfg(self) -> ControlFlowGraph:
        """Provide the raw control flow graph from the code object.
        Can be used to instrument the control flow.

        Returns:
            The raw control-flow graph from the code object
        """
        return self._bytecode_cfg

    @staticmethod
    def reverse(cfg: CFG) -> CFG:
        """Reverses a control-flow graph, i.e., entry nodes become exit nodes and
        vice versa.

        Args:
            cfg: The control-flow graph to reverse

        Returns:
            The reversed control-flow graph
        """
        reversed_cfg = CFG(cfg.bytecode_cfg())
        # pylint: disable=attribute-defined-outside-init
        reversed_cfg._graph = cfg._graph.reverse(copy=True)
        return reversed_cfg

    def reversed(self) -> CFG:
        """Provides the reversed graph of this graph.

        Returns:
            The reversed graph
        """
        return CFG.reverse(self)

    @staticmethod
    def copy_graph(cfg: CFG) -> CFG:
        """Provides a copy of the control-flow graph.

        Args:
            cfg: The original graph

        Returns:
            The copied graph
        """
        copy = CFG(
            ControlFlowGraph()
        )  # TODO(fk) Cloning the bytecode cfg is complicated.
        # pylint: disable=attribute-defined-outside-init
        copy._graph = cfg._graph.copy()
        return copy

    def copy(self) -> CFG:
        """Provides a copy of the control-flow graph.

        Returns:
            The copied graph
        """
        return CFG.copy_graph(self)

    @staticmethod
    def _create_nodes(
        blocks: ControlFlowGraph,
    ) -> Tuple[Dict[int, List[int]], Dict[int, ProgramGraphNode]]:
        nodes: Dict[int, ProgramGraphNode] = {}
        edges: Dict[int, List[int]] = {}
        for node_index, block in enumerate(blocks):
            node = ProgramGraphNode(index=node_index, basic_block=block)
            nodes[node_index] = node
            if node_index not in edges:
                edges[node_index] = []

            next_block = block.next_block
            if next_block:
                next_index = blocks.get_block_index(next_block)
                edges[node_index].append(next_index)
            if target_block := block.get_jump():
                next_index = blocks.get_block_index(target_block)
                edges[node_index].append(next_index)
        return edges, nodes

    @staticmethod
    def _create_graph(
        cfg: CFG, edges: Dict[int, List[int]], nodes: Dict[int, ProgramGraphNode]
    ):
        # add nodes to graph
        for node in nodes.values():
            cfg.add_node(node)

        # add edges to graph
        for predecessor in edges.keys():
            successors = edges.get(predecessor)
            for successor in cast(List[int], successors):
                predecessor_node = nodes.get(predecessor)
                successor_node = nodes.get(successor)
                assert predecessor_node
                assert successor_node
                cfg.add_edge(predecessor_node, successor_node)

    @staticmethod
    def _insert_dummy_entry_node(cfg: CFG) -> CFG:
        dummy_entry_node = ProgramGraphNode(index=-1, is_artificial=True)
        entry_node = cfg.entry_node
        if not entry_node and len(cfg.nodes) == 1:
            entry_node = cfg.nodes.pop()  # There is only one candidate to pop
        elif not entry_node:
            entry_node = [n for n in cfg.nodes if n.index == 0][0]
        cfg.add_node(dummy_entry_node)
        cfg.add_edge(dummy_entry_node, entry_node)
        return cfg

    @staticmethod
    def _insert_dummy_exit_node(cfg: CFG) -> CFG:
        dummy_exit_node = ProgramGraphNode(index=sys.maxsize, is_artificial=True)
        exit_nodes = cfg.exit_nodes
        if not exit_nodes and len(cfg.nodes) == 1:
            exit_nodes = cfg.nodes
        cfg.add_node(dummy_exit_node)
        for exit_node in exit_nodes:
            cfg.add_edge(exit_node, dummy_exit_node)
        return cfg

    @property
    def cyclomatic_complexity(self) -> int:
        """Calculates McCabe's cyclomatic complexity for this control-flow graph

        Returns:
            McCabe's cyclocmatic complexity number
        """
        return len(self._graph.edges) - len(self._graph.nodes) + 2

    @property
    def diameter(self) -> int:
        """Computes the diameter of the graph

        Returns:
            The diameter of the graph
        """
        if self._diameter is None:
            # Do this computation lazily
            try:
                self._diameter = nx.diameter(self._graph, usebounds=True)
            except nx.NetworkXError:
                # It seems like NetworkX makes some assumptions on the graph which
                # are not documented (or which I could not find at least) that caused
                # these errors.
                # If the diameter computation fails for some reason, use an upper bound
                self._diameter = len(self._graph.edges)
        return self._diameter


class DominatorTree(ProgramGraph[ProgramGraphNode]):
    """Implements a dominator tree."""

    @staticmethod
    def compute(graph: CFG) -> DominatorTree:
        """Computes the dominator tree for a control-flow graph.

        Args:
            graph: The control-flow graph

        Returns:
            The dominator tree for the control-flow graph
        """
        return DominatorTree.compute_dominance_tree(graph)

    @staticmethod
    def compute_post_dominator_tree(graph: CFG) -> DominatorTree:
        """Computes the post-dominator tree for a control-flow graph.

        Args:
            graph: The control-flow graph

        Returns:
            The post-dominator tree for the control-flow graph
        """
        reversed_cfg = graph.reversed()
        return DominatorTree.compute(reversed_cfg)

    @staticmethod
    def compute_dominance_tree(graph: CFG) -> DominatorTree:
        """Computes the dominance tree for a control-flow graph.

        Args:
            graph: The control-flow graph

        Returns:
            The dominance tree for the control-flow graph
        """
        dominance: Dict[
            ProgramGraphNode, Set[ProgramGraphNode]
        ] = DominatorTree._calculate_dominance(graph)
        for dominance_node, nodes in dominance.items():
            nodes.discard(dominance_node)
        dominance_tree = DominatorTree()
        entry_node = graph.entry_node
        assert entry_node is not None
        dominance_tree.add_node(entry_node)

        node_queue: queue.SimpleQueue = queue.SimpleQueue()
        node_queue.put(entry_node)
        while not node_queue.empty():
            node: ProgramGraphNode = node_queue.get()
            for current, dominators in dominance.items():
                if node in dominators:
                    dominators.remove(node)
                    if len(dominators) == 0:
                        dominance_tree.add_node(current)
                        dominance_tree.add_edge(node, current)
                        node_queue.put(current)
        return dominance_tree

    @staticmethod
    def _calculate_dominance(
        graph: CFG,
    ) -> Dict[ProgramGraphNode, Set[ProgramGraphNode]]:
        dominance_map: Dict[ProgramGraphNode, Set[ProgramGraphNode]] = {}
        entry = graph.entry_node
        assert entry, "Cannot work with a graph without entry nodes"
        entry_dominators: Set[ProgramGraphNode] = {entry}
        dominance_map[entry] = entry_dominators

        for node in graph.nodes:
            if node == entry:
                continue
            all_nodes: Set[ProgramGraphNode] = set(graph.nodes)
            dominance_map[node] = all_nodes

        changed: bool = True
        while changed:
            changed = False
            for node in graph.nodes:
                if node == entry:
                    continue
                current_dominators = dominance_map.get(node)
                new_dominators = DominatorTree._calculate_dominators(
                    graph, dominance_map, node
                )

                if current_dominators != new_dominators:
                    changed = True
                    dominance_map[node] = new_dominators
                    break

        return dominance_map

    @staticmethod
    def _calculate_dominators(
        graph: CFG,
        dominance_map: Dict[ProgramGraphNode, Set[ProgramGraphNode]],
        node: ProgramGraphNode,
    ) -> Set[ProgramGraphNode]:
        dominators: Set[ProgramGraphNode] = {node}
        intersection: Set[ProgramGraphNode] = set()
        predecessors = graph.get_predecessors(node)
        if not predecessors:
            return set()

        first_time: bool = True
        for predecessor in predecessors:
            predecessor_dominators = dominance_map.get(predecessor)
            assert predecessor_dominators is not None, "Cannot be None"
            if first_time:
                intersection = intersection.union(predecessor_dominators)
                first_time = False
            else:
                intersection.intersection_update(predecessor_dominators)
        intersection = intersection.union(dominators)
        return intersection


# pylint:disable=too-few-public-methods.
class ControlDependenceGraph(ProgramGraph[ProgramGraphNode]):
    """Implements a control-dependence graph."""

    @staticmethod
    def compute(graph: CFG) -> ControlDependenceGraph:
        """Computes the control-dependence graph for a given control-flow graph.

        Args:
            graph: The control-flow graph

        Returns:
            The control-dependence graph
        """
        augmented_cfg = ControlDependenceGraph._create_augmented_graph(graph)
        post_dominator_tree = DominatorTree.compute_post_dominator_tree(augmented_cfg)
        cdg = ControlDependenceGraph()
        nodes = augmented_cfg.nodes

        for node in nodes:
            cdg.add_node(node)

        # Find matching edges in the CFG.
        edges: Set[ControlDependenceGraph._Edge] = set()
        for source in nodes:
            for target in augmented_cfg.get_successors(source):
                if source not in post_dominator_tree.get_transitive_successors(target):
                    edges.add(
                        ControlDependenceGraph._Edge(source=source, target=target)
                    )

        # Mark nodes in the PDT and construct edges for them.
        for edge in edges:
            least_common_ancestor = post_dominator_tree.get_least_common_ancestor(
                edge.source, edge.target
            )
            current = edge.target
            while current != least_common_ancestor:
                cdg.add_edge(edge.source, current)
                predecessors = post_dominator_tree.get_predecessors(current)
                assert len(predecessors) == 1, (
                    "Cannot have more than one predecessor in a tree, this violates a "
                    + "tree invariant"
                )
                current = predecessors.pop()

            if least_common_ancestor is edge.source:
                cdg.add_edge(edge.source, least_common_ancestor)

        return filter_dead_code_nodes(cdg, entry_node_index=-sys.maxsize)

    @staticmethod
    def _create_augmented_graph(graph: CFG) -> CFG:
        entry_node = graph.entry_node
        assert entry_node, "Cannot work with CFG without entry node"
        exit_nodes = graph.exit_nodes
        augmented_graph = graph.copy()
        start_node = ProgramGraphNode(index=-sys.maxsize, is_artificial=True)
        augmented_graph.add_node(start_node)
        augmented_graph.add_edge(start_node, entry_node)
        for exit_node in exit_nodes:
            augmented_graph.add_edge(start_node, exit_node)
        return augmented_graph

    @dataclass(eq=True, frozen=True)
    class _Edge:
        source: ProgramGraphNode
        target: ProgramGraphNode
