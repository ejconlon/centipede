"""Pattern DAG representation with union-find for efficient equality checking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NewType, Optional, Tuple

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
from spiny.common import Box
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
        return bool(self.root_node_id() == other.root_node_id())

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


def _hash_pattern[T](pat_f: PatF[T, PatFind]) -> PatHash:
    """Create a canonical tuple representation of a pattern."""
    canon: tuple[object, ...]
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


def _next_pat_find(
    id_src: Box[PatId],
) -> PatFind:
    """Add a new node to the DAG and return its Find handle."""
    node_id = id_src.value
    id_src.value = PatId(node_id + 1)
    return Find(node_id)


def _convert_pat_node[T](
    id_src: Box[PatId], nodes: Dict[PatId, PatNode[T]], root_pat: Pat[T]
) -> PatFind:
    """Convert a Pat tree to DAG representation using iterative approach with stack.

    Creates a new node for every pattern - canonicalization will merge duplicates later.
    Returns the Find handle for the root of the converted DAG.
    """
    # Stack of (pat, find) pairs to process
    stack: List[Tuple[Pat[T], PatFind]] = []

    # Allocate ID for root and push to stack
    root_find = _next_pat_find(id_src)
    stack.append((root_pat, root_find))

    while stack:
        pat, find = stack.pop()

        patf: PatF[T, PatFind]
        match pat.unwrap:
            case PatSilent():
                patf = PatSilent()

            case PatPure(value):
                patf = PatPure(value)

            case PatSeq(pats):
                child_finds = []
                for child in pats.iter():
                    child_find = _next_pat_find(id_src)
                    child_finds.append(child_find)
                    stack.append((child, child_find))
                converted_seq = PSeq.mk(child_finds)
                patf = PatSeq(converted_seq)

            case PatPar(pats):
                child_finds = []
                for child in pats.iter():
                    child_find = _next_pat_find(id_src)
                    child_finds.append(child_find)
                    stack.append((child, child_find))
                converted_seq = PSeq.mk(child_finds)
                patf = PatPar(converted_seq)

            case PatRand(pats):
                child_finds = []
                for child in pats.iter():
                    child_find = _next_pat_find(id_src)
                    child_finds.append(child_find)
                    stack.append((child, child_find))
                converted_seq = PSeq.mk(child_finds)
                patf = PatRand(converted_seq)

            case PatAlt(pats):
                child_finds = []
                for child in pats.iter():
                    child_find = _next_pat_find(id_src)
                    child_finds.append(child_find)
                    stack.append((child, child_find))
                converted_seq = PSeq.mk(child_finds)
                patf = PatAlt(converted_seq)

            case PatPoly(pats, subdiv):
                child_finds = []
                for child in pats.iter():
                    child_find = _next_pat_find(id_src)
                    child_finds.append(child_find)
                    stack.append((child, child_find))
                converted_seq = PSeq.mk(child_finds)
                patf = PatPoly(converted_seq, subdiv)

            case PatEuc(child, hits, steps, rotation):
                child_find = _next_pat_find(id_src)
                stack.append((child, child_find))
                patf = PatEuc(child_find, hits, steps, rotation)

            case PatSpeed(child, op, factor):
                child_find = _next_pat_find(id_src)
                stack.append((child, child_find))
                patf = PatSpeed(child_find, op, factor)

            case PatStretch(child, count):
                child_find = _next_pat_find(id_src)
                stack.append((child, child_find))
                patf = PatStretch(child_find, count)

            case PatProb(child, chance):
                child_find = _next_pat_find(id_src)
                stack.append((child, child_find))
                patf = PatProb(child_find, chance)

            case PatRepeat(child, count):
                child_find = _next_pat_find(id_src)
                stack.append((child, child_find))
                patf = PatRepeat(child_find, count)

            case _:
                raise PartialMatchException(f"Unknown pattern type: {type(pat.unwrap)}")

        nodes[find._node_id] = PatNode(find, patf)

    return root_find


class PatDag[T]:
    """Pattern DAG representation with union-find for equality."""

    def __init__(
        self, id_src: Box[PatId], root_find: PatFind, nodes: Dict[PatId, PatNode[T]]
    ) -> None:
        self._id_src = id_src
        self._root_find = root_find
        self._nodes = nodes

    def postorder(self) -> List[PatId]:
        """Return nodes in postorder (bottom-up) traversal from the root.

        Children are visited before their parents, leaves first, root last.
        Only returns reachable nodes. Uses explicit stack to avoid recursion.
        """
        visited = set()
        result = []

        # Stack of (node_id, children_processed)
        stack = [(self._root_find.root_node_id(), False)]

        while stack:
            node_id, children_processed = stack.pop()

            if node_id in visited or node_id not in self._nodes:
                continue

            if children_processed:
                # Children have been processed, now process this node
                visited.add(node_id)
                result.append(node_id)
            else:
                # Mark this node for processing after children
                stack.append((node_id, True))

                # Add children to stack (they'll be processed first due to stack order)
                pat_node = self._nodes[node_id]
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
        unreachable = [k for k in self._nodes if k not in reachable]
        for k in unreachable:
            del self._nodes[k]

        return len(unreachable) > 0

    def has_node(self, node_id: PatId) -> bool:
        """Check if a node exists in the DAG."""
        return node_id in self._nodes

    def get_pat_node(self, node_id: PatId) -> PatNode[T]:
        """Get the PatNode for a given node ID."""
        return self._nodes[node_id]

    def set_pat_node(self, node_id: PatId, pat_node: PatNode[T]) -> None:
        """Set the PatNode for a given node ID."""
        self._nodes[node_id] = pat_node

    def get_node(self, find: PatFind) -> PatF[T, PatFind]:
        """Get the pattern node associated with a Find handle."""
        return self._nodes[find.root_node_id()].patf

    def add_node(self, pat: PatF[T, PatFind]) -> PatFind:
        """Add a new node to the DAG and return its Find handle."""
        node_id = self._id_src.value
        self._id_src.value = PatId(node_id + 1)
        find = Find(node_id)
        pat_node = PatNode(find, pat)
        self._nodes[node_id] = pat_node
        return find

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

        # Track nodes that become garbage during canonicalization
        garbage: set[PatId] = set()

        for _ in range(max_iterations):
            changed = False
            # Build a mapping from pattern hash to Find nodes
            pattern_map: Dict[PatHash, PatFind] = {}

            # Process nodes in postorder - children before parents
            for node_id in node_order:
                # Skip nodes that have become garbage
                if node_id in garbage:
                    continue

                pat_node = self._nodes[node_id]
                find = pat_node.find

                # Create a canonical hash of the pattern
                # Since we're processing bottom-up, all children are already canonicalized
                canon_hash = _hash_pattern(pat_node.patf)

                if canon_hash in pattern_map:
                    # Found equivalent pattern, merge them
                    existing = pattern_map[canon_hash]
                    if existing.root_node_id() != find.root_node_id():
                        # Before union, determine which nodes will become non-roots
                        find_root_id = find.root_node_id()
                        existing_root_id = existing.root_node_id()

                        find.union(existing)

                        # After union, the node that's no longer the root becomes garbage
                        new_root_id = find.root_node_id()
                        if find_root_id != new_root_id:
                            garbage.add(find_root_id)
                        if existing_root_id != new_root_id:
                            garbage.add(existing_root_id)

                        changed = True
                        any_changes = True
                else:
                    pattern_map[canon_hash] = find

            # If no changes were made, we've reached fixed point
            if not changed:
                break

        # Clean up garbage nodes at the end
        for node_id in garbage:
            del self._nodes[node_id]

        return any_changes

    @staticmethod
    def from_pat(pat: Pat[T]) -> PatDag[T]:
        """Convert a Pat tree structure to a PatDag."""
        id_src = Box(PatId(0))
        nodes: Dict[PatId, PatNode[T]] = {}
        root_find = _convert_pat_node(id_src, nodes, pat)
        return PatDag(id_src, root_find, nodes)

    def to_pat(self) -> Pat[T]:
        """Convert this PatDag back to a Pat tree structure.

        Processes nodes in postorder to ensure children are converted before parents.
        """
        cache: Dict[PatId, Pat[T]] = {}

        # Process nodes in postorder - children before parents
        for node_id in self.postorder():
            patf = self._nodes[node_id].patf

            match patf:
                case PatSilent():
                    result = Pat(PatSilent())
                case PatPure(value):
                    result = Pat(PatPure(value))
                case PatSeq(pats):
                    # All children are guaranteed to be in cache
                    converted = [cache[child.root_node_id()] for child in pats.iter()]
                    result = Pat.seq(converted)
                case PatPar(pats):
                    converted = [cache[child.root_node_id()] for child in pats.iter()]
                    result = Pat.par(converted)
                case PatRand(pats):
                    converted = [cache[child.root_node_id()] for child in pats.iter()]
                    result = Pat.rand(converted)
                case PatAlt(pats):
                    converted = [cache[child.root_node_id()] for child in pats.iter()]
                    result = Pat.alt(converted)
                case PatEuc(child, hits, steps, rotation):
                    converted_child = cache[child.root_node_id()]
                    result = Pat(PatEuc(converted_child, hits, steps, rotation))
                case PatPoly(pats, subdiv):
                    converted = [cache[child.root_node_id()] for child in pats.iter()]
                    result = Pat.poly(converted, subdiv)
                case PatSpeed(child, op, factor):
                    converted_child = cache[child.root_node_id()]
                    result = Pat(PatSpeed(converted_child, op, factor))
                case PatStretch(child, count):
                    converted_child = cache[child.root_node_id()]
                    result = Pat(PatStretch(converted_child, count))
                case PatProb(child, chance):
                    converted_child = cache[child.root_node_id()]
                    result = Pat(PatProb(converted_child, chance))
                case PatRepeat(child, count):
                    converted_child = cache[child.root_node_id()]
                    result = Pat(PatRepeat(converted_child, count))
                case _:
                    raise PartialMatchException(f"Unknown pattern type: {type(patf)}")

            cache[node_id] = result

        # Return the root pattern
        return cache[self._root_find.root_node_id()]
