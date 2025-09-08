"""Pattern DAG representation with union-find for efficient equality checking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, NewType, Optional

from minipat.common import PartialMatchException
from minipat.pat import (
    Pat,
    PatAlt,
    PatEuc,
    PatF,
    PatPar,
    PatPoly,
    PatProb,
    PatPure,
    PatRand,
    PatRepeat,
    PatSeq,
    PatSilent,
    PatSpeed,
    PatStretch,
)
from spiny.seq import PSeq


class Find[I]:
    """Union-find node with path compression for efficient equality checks."""

    def __init__(self, node_id: I):
        self._node_id = node_id
        self._parent: Optional[Find[I]] = None

    def root(self) -> Find[I]:
        """Find root with path compression."""
        if self._parent is None:
            return self

        # Path compression: compress all ancestors to the root
        path = []
        current = self
        while current._parent is not None:
            path.append(current)
            current = current._parent

        # Make all nodes in path point directly to root
        for node in path:
            node._parent = current

        return current

    def is_root(self) -> bool:
        """Check if this node is a root (has no parent)."""
        return self._parent is None

    def root_node_id(self) -> I:
        """Get the node ID of the root of this Find node."""
        return self.root()._node_id

    def union(self, other: Find[I]) -> None:
        """Union two sets by making one root point to the other."""
        root1 = self.root()
        root2 = other.root()

        if root1 != root2:
            # Make root2 the parent of root1
            root1._parent = root2

    def __eq__(self, other: object) -> bool:
        """Two Find nodes are equal if they have the same root."""
        if not isinstance(other, Find):
            return False
        return self.root_node_id() == other.root_node_id()

    def __hash__(self) -> int:
        """Hash based on root's node_id for use in sets/dicts."""
        return hash(self.root_node_id())

    def __repr__(self) -> str:
        return f"Find({self._node_id}, root={self.root_node_id()})"


# Type definitions
PatId = NewType("PatId", int)
PatHash = NewType("PatHash", int)

# Type alias for Find specialized with PatId
type PatFind = Find[PatId]


@dataclass(frozen=True)
class PatNode[T]:
    """A pattern node bundling a Find handle with its pattern functor."""

    find: PatFind
    patf: PatF[T, PatFind]


@dataclass
class PatDag[T]:
    """Pattern DAG representation with union-find for equality."""

    id_src: int = field(default=0)
    root: PatId = field(default=PatId(0))
    nodes: Dict[PatId, PatNode[T]] = field(default_factory=dict)

    def _next_id(self) -> PatId:
        """Generate next unique node ID."""
        self.id_src += 1
        return PatId(self.id_src)

    def postorder(self) -> List[PatId]:
        """Return nodes in postorder (bottom-up) traversal from the root.

        Children are visited before their parents, leaves first, root last.
        Only returns reachable nodes. Uses explicit stack to avoid recursion.
        """
        visited = set()
        result = []

        # Stack of (node_id, children_processed)
        stack = [(self.root, False)]

        while stack:
            node_id, children_processed = stack.pop()

            if node_id in visited or node_id not in self.nodes:
                continue

            if children_processed:
                # Children have been processed, now process this node
                visited.add(node_id)
                result.append(node_id)
            else:
                # Mark this node for processing after children
                stack.append((node_id, True))

                # Add children to stack (they'll be processed first due to stack order)
                pat_node = self.nodes[node_id]
                match pat_node.patf:
                    case (
                        PatSeq(pats)
                        | PatPar(pats)
                        | PatRand(pats)
                        | PatAlt(pats)
                        | PatPoly(pats, _)
                    ):
                        for child in pats.iter():
                            stack.append((child.root_node_id(), False))
                    case (
                        PatEuc(child, _, _, _)
                        | PatSpeed(child, _, _)
                        | PatStretch(child, _)
                        | PatProb(child, _)
                        | PatRepeat(child, _)
                    ):
                        stack.append((child.root_node_id(), False))
                    case PatSilent() | PatPure(_):
                        pass

        return result

    def collect(self) -> bool:
        """Garbage collect unused nodes.
        Leaves only union-find root nodes reachable from the dag root.

        Returns:
            True if any nodes were removed, False otherwise.
        """
        # Get all reachable nodes using postorder traversal
        reachable = set(self.postorder())

        # Remove unreachable nodes from existing dict
        unreachable = [k for k in self.nodes if k not in reachable]
        for k in unreachable:
            del self.nodes[k]

        return len(unreachable) > 0

    def add_node(self, pat: PatF[T, PatFind]) -> PatFind:
        """Add a new node to the DAG and return its Find handle."""
        node_id = self._next_id()
        find = Find(node_id)
        pat_node = PatNode(find, pat)
        self.nodes[node_id] = pat_node
        return find

    def get_node(self, find: PatFind) -> PatF[T, PatFind]:
        """Get the pattern node associated with a Find handle."""
        return self.nodes[find.root_node_id()].patf

    def merge_equivalent(self, find1: PatFind, find2: PatFind) -> None:
        """Merge two equivalent patterns using union-find."""
        find1.union(find2)

    def canonicalize(self, max_iterations: int = 100) -> bool:
        """Canonicalize the DAG by finding and merging equivalent subpatterns.

        Processes nodes in postorder (bottom-up) for most efficient equivalence
        closure calculation. May iterate multiple times until fixed point.
        Collects garbage once at the end if any changes were made.

        Returns:
            True if any patterns were merged, False otherwise.
        """
        any_changes = False

        # Calculate postorder once - the topological order doesn't change during canonicalization
        node_order = self.postorder()

        for _ in range(max_iterations):
            changed = False
            # Build a mapping from pattern hash to Find nodes
            pattern_map: Dict[PatHash, PatFind] = {}

            # Process nodes in postorder - children before parents
            for node_id in node_order:
                pat_node = self.nodes[node_id]
                find = pat_node.find

                # Create a canonical hash of the pattern
                # Since we're processing bottom-up, all children are already canonicalized
                canon_hash = self._hash_pattern(pat_node.patf)

                if canon_hash in pattern_map:
                    # Found equivalent pattern, merge them
                    existing = pattern_map[canon_hash]
                    if existing.root_node_id() != find.root_node_id():
                        find.union(existing)
                        changed = True
                        any_changes = True
                else:
                    pattern_map[canon_hash] = find

            # If no changes were made, we've reached fixed point
            if not changed:
                break

        # Collect garbage once at the end if any changes were made
        if any_changes:
            self.collect()

        return any_changes

    def _hash_pattern(self, pat_f: PatF[T, PatFind]) -> PatHash:
        """Create a canonical tuple representation of a pattern."""
        canon: tuple
        match pat_f:
            case PatSilent():
                canon = ("silent",)
            case PatPure(value):
                canon = ("pure", value)
            case PatSeq(pats):
                canon = ("seq", tuple(find.root_node_id() for find in pats.iter()))
            case PatPar(pats):
                canon = ("par", tuple(find.root_node_id() for find in pats.iter()))
            case PatRand(pats):
                canon = ("rand", tuple(find.root_node_id() for find in pats.iter()))
            case PatAlt(pats):
                canon = ("alt", tuple(find.root_node_id() for find in pats.iter()))
            case PatEuc(child, hits, steps, rotation):
                canon = ("euc", child.root_node_id(), hits, steps, rotation)
            case PatPoly(pats, subdiv):
                canon = (
                    "poly",
                    tuple(find.root_node_id() for find in pats.iter()),
                    subdiv,
                )
            case PatSpeed(child, op, factor):
                canon = ("speed", child.root_node_id(), op, factor)
            case PatStretch(child, count):
                canon = ("stretch", child.root_node_id(), count)
            case PatProb(child, chance):
                canon = ("prob", child.root_node_id(), chance)
            case PatRepeat(child, count):
                canon = ("repeat", child.root_node_id(), count)
            case _:
                raise PartialMatchException(f"Unknown pattern type: {type(pat_f)}")
        return PatHash(hash(canon))

    @staticmethod
    def _convert_pat_node(
        p: Pat, dag: PatDag[T], node_map: Dict[int, PatFind]
    ) -> PatFind:
        """Convert a single Pat node to DAG representation."""
        # Check if we've already converted this node (for sharing)
        pat_id = id(p)
        if pat_id in node_map:
            return node_map[pat_id]

        match p.unwrap:
            case PatSilent():
                find = dag.add_node(PatSilent())
            case PatPure(value):
                find = dag.add_node(PatPure(value))
            case PatSeq(pats):
                converted = PSeq.mk(
                    [
                        PatDag._convert_pat_node(child, dag, node_map)
                        for child in pats.iter()
                    ]
                )
                find = dag.add_node(PatSeq(converted))
            case PatPar(pats):
                converted = PSeq.mk(
                    [
                        PatDag._convert_pat_node(child, dag, node_map)
                        for child in pats.iter()
                    ]
                )
                find = dag.add_node(PatPar(converted))
            case PatRand(pats):
                converted = PSeq.mk(
                    [
                        PatDag._convert_pat_node(child, dag, node_map)
                        for child in pats.iter()
                    ]
                )
                find = dag.add_node(PatRand(converted))
            case PatAlt(pats):
                converted = PSeq.mk(
                    [
                        PatDag._convert_pat_node(child, dag, node_map)
                        for child in pats.iter()
                    ]
                )
                find = dag.add_node(PatAlt(converted))
            case PatEuc(child, hits, steps, rotation):
                converted_child = PatDag._convert_pat_node(child, dag, node_map)
                find = dag.add_node(PatEuc(converted_child, hits, steps, rotation))
            case PatPoly(pats, subdiv):
                converted = PSeq.mk(
                    [
                        PatDag._convert_pat_node(child, dag, node_map)
                        for child in pats.iter()
                    ]
                )
                find = dag.add_node(PatPoly(converted, subdiv))
            case PatSpeed(child, op, factor):
                converted_child = PatDag._convert_pat_node(child, dag, node_map)
                find = dag.add_node(PatSpeed(converted_child, op, factor))
            case PatStretch(child, count):
                converted_child = PatDag._convert_pat_node(child, dag, node_map)
                find = dag.add_node(PatStretch(converted_child, count))
            case PatProb(child, chance):
                converted_child = PatDag._convert_pat_node(child, dag, node_map)
                find = dag.add_node(PatProb(converted_child, chance))
            case PatRepeat(child, count):
                converted_child = PatDag._convert_pat_node(child, dag, node_map)
                find = dag.add_node(PatRepeat(converted_child, count))
            case _:
                raise PartialMatchException(f"Unknown pattern type: {type(p.unwrap)}")

        node_map[pat_id] = find
        return find

    @staticmethod
    def from_pat(pat: Pat[T]) -> PatDag[T]:
        """Convert a Pat tree structure to a PatDag."""
        dag = PatDag[T]()
        node_map: Dict[int, PatFind] = {}
        root_find = PatDag._convert_pat_node(pat, dag, node_map)
        dag.root = root_find._node_id  # This is the actual node ID, not root
        return dag

    def _convert_find_node(self, find: PatFind, cache: Dict[PatId, Pat[T]]) -> Pat[T]:
        """Convert a Find node back to a Pat."""
        root_id = find.root_node_id()

        if root_id in cache:
            return cache[root_id]

        pat_f = self.nodes[root_id].patf

        match pat_f:
            case PatSilent():
                result = Pat(PatSilent())
            case PatPure(value):
                result = Pat(PatPure(value))
            case PatSeq(pats):
                converted = [
                    self._convert_find_node(child, cache) for child in pats.iter()
                ]
                result = Pat.seq(converted)
            case PatPar(pats):
                converted = [
                    self._convert_find_node(child, cache) for child in pats.iter()
                ]
                result = Pat.par(converted)
            case PatRand(pats):
                converted = [
                    self._convert_find_node(child, cache) for child in pats.iter()
                ]
                result = Pat.rand(converted)
            case PatAlt(pats):
                converted = [
                    self._convert_find_node(child, cache) for child in pats.iter()
                ]
                result = Pat.alt(converted)
            case PatEuc(child, hits, steps, rotation):
                converted_child = self._convert_find_node(child, cache)
                result = Pat(PatEuc(converted_child, hits, steps, rotation))
            case PatPoly(pats, subdiv):
                converted = [
                    self._convert_find_node(child, cache) for child in pats.iter()
                ]
                result = Pat.poly(converted, subdiv)
            case PatSpeed(child, op, factor):
                converted_child = self._convert_find_node(child, cache)
                result = Pat(PatSpeed(converted_child, op, factor))
            case PatStretch(child, count):
                converted_child = self._convert_find_node(child, cache)
                result = Pat(PatStretch(converted_child, count))
            case PatProb(child, chance):
                converted_child = self._convert_find_node(child, cache)
                result = Pat(PatProb(converted_child, chance))
            case PatRepeat(child, count):
                converted_child = self._convert_find_node(child, cache)
                result = Pat(PatRepeat(converted_child, count))
            case _:
                raise PartialMatchException(f"Unknown pattern type: {type(pat_f)}")

        cache[root_id] = result
        return result

    def to_pat(self) -> Pat[T]:
        """Convert this PatDag back to a Pat tree structure."""
        cache: Dict[PatId, Pat[T]] = {}
        root_find = Find[PatId](self.root)
        return self._convert_find_node(root_find, cache)
