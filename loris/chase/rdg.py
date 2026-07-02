"""
chase_inference/rdg.py
-----------------------
Rule Dependency Graph (RDG) for RILL influence estimation and trust checking.

Static analysis of rule-to-rule dependencies: rule A → rule B if A's
consequence label appears in B's body as a LabelPredicate.  Built once,
reused throughout the entire RILL active-learning loop.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set

from loris.predicates import LabelPredicate


class RuleDependencyGraph:
    """Directed graph encoding trigger dependencies between rules.

    Parameters
    ----------
    rules : List
        List of RDL rule objects (from ``rule_discovery``).
    label_names : List[str]
        Ordered label vocabulary.
    """

    def __init__(self, rules: list, label_names: List[str]) -> None:
        self.rules = list(rules)
        self.label_names = list(label_names)
        self.n_rules = len(self.rules)

        # label τ → list of rule indices whose body contains LabelPredicate(τ)
        self.label_to_consumers: Dict[str, List[int]] = defaultdict(list)
        # rule_idx → set of downstream rule indices triggered by its consequence
        self.adjacency: Dict[int, Set[int]] = defaultdict(set)
        # rule_idx → which labels this rule can produce
        self.rule_to_consequence: Dict[int, str] = {}
        # label → list of rule indices that produce this label
        self.label_to_producers: Dict[str, List[int]] = defaultdict(list)

        self._build()

    def _build(self) -> None:
        """Construct the dependency graph."""
        # Step 1: scan all rules for LabelPredicate in body
        for idx, rule in enumerate(self.rules):
            self.rule_to_consequence[idx] = rule.consequence
            self.label_to_producers[rule.consequence].append(idx)

            for pred in rule.body:
                if isinstance(pred, LabelPredicate):
                    self.label_to_consumers[pred.label].append(idx)

        # Step 2: build adjacency — rule A → rules that consume A's consequence
        for idx, rule in enumerate(self.rules):
            downstream = self.label_to_consumers.get(rule.consequence, [])
            self.adjacency[idx] = set(downstream)

    def rules_depending_on_label(self, label: str) -> List[int]:
        """Return rule indices whose body contains ``LabelPredicate(label)``."""
        return list(self.label_to_consumers.get(label, []))

    def rules_producing_label(self, label: str) -> List[int]:
        """Return rule indices whose consequence is ``label``."""
        return list(self.label_to_producers.get(label, []))

    def bfs_from_label(self, label: str, max_depth: int = -1) -> Set[int]:
        """BFS from all rules that produce ``label``, returning reachable rule set.

        Parameters
        ----------
        label : str
            Starting label.
        max_depth : int
            Maximum BFS depth (-1 = unlimited).

        Returns
        -------
        Set[int]
            Set of reachable rule indices (includes producers of ``label``).
        """
        seeds = self.label_to_producers.get(label, [])
        # Also include rules that *consume* this label (direct dependents)
        consumers = self.label_to_consumers.get(label, [])
        start = set(seeds) | set(consumers)
        return self._bfs(start, max_depth)

    def bfs_from_rule(self, rule_idx: int, max_depth: int = -1) -> Set[int]:
        """BFS from a single rule, returning all downstream reachable rules."""
        return self._bfs({rule_idx}, max_depth)

    def _bfs(self, start: Set[int], max_depth: int) -> Set[int]:
        """Generic BFS over the adjacency graph."""
        visited: Set[int] = set(start)
        frontier: deque = deque((r, 0) for r in start)

        while frontier:
            rule_idx, depth = frontier.popleft()
            if max_depth >= 0 and depth >= max_depth:
                continue
            for downstream in self.adjacency.get(rule_idx, set()):
                if downstream not in visited:
                    visited.add(downstream)
                    frontier.append((downstream, depth + 1))

        return visited

    def __repr__(self) -> str:
        n_edges = sum(len(v) for v in self.adjacency.values())
        return (
            f"RuleDependencyGraph(n_rules={self.n_rules}, "
            f"n_edges={n_edges}, "
            f"n_label_consumers={sum(len(v) for v in self.label_to_consumers.values())})"
        )
