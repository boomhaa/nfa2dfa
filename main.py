#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import argparse
import copy
import logging
from dataclasses import dataclass, field
from typing import Set, Dict, Any, FrozenSet, List

EPS = "ε"  # epsilon symbol

logger = logging.getLogger(__name__)


@dataclass
class NFA:
    states: Set[Any]
    alphabet: Set[str]
    start_state: Any
    accept_states: Set[Any]
    # transitions[state][symbol] = {next_states}
    transitions: Dict[Any, Dict[str, Set[Any]]]


@dataclass
class DFA:
    states: Set[int]
    alphabet: Set[str]
    start_state: int
    accept_states: Set[int]
    transitions: Dict[int, Dict[str, int]]
    state_mapping: Dict[int, FrozenSet[Any]] = field(default_factory=dict)


class DotParseError(Exception):
    pass


def parse_dot_nfa(path: str) -> NFA:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError as e:
        raise DotParseError(f"File not found: {path}") from e
    except OSError as e:
        raise DotParseError(f"Error reading file '{path}': {e}") from e

    try:
        edge_re = re.compile(r'(\w+)\s*->\s*(\w+)\s*\[label="([^"]*)"];')
        accept_re = re.compile(r'(\w+)\s*\[([^]]*shape\s*=\s*doublecircle[^]]*)];')
        start_re = re.compile(r'__start\s*->\s*(\w+)\s*;')
    except re.error as e:
        raise DotParseError(f"Regex compilation error: {e}") from e

    transitions: Dict[str, Dict[str, Set[str]]] = {}
    states: Set[str] = set()
    alphabet: Set[str] = set()
    accept_states: Set[str] = set()
    start_state: str | None = None

    # start state
    m = start_re.search(text)
    if m:
        start_state = m.group(1)

    # accepting states
    for m in accept_re.finditer(text):
        q = m.group(1)
        if q != "__start":
            accept_states.add(q)
            states.add(q)

    # edges
    for m in edge_re.finditer(text):
        u, v, label = m.groups()
        if u == "__start":
            # this is the start arrow, not NFA transition
            continue
        states.add(u)
        states.add(v)

        parts = [p.strip() for p in label.split(",")]
        for sym in parts:
            if not sym:
                continue
            transitions.setdefault(u, {}).setdefault(sym, set()).add(v)
            if sym != EPS:
                alphabet.add(sym)

    if start_state is None:
        raise DotParseError("Start state not found (__start -> q) in DOT file")

    def convert_state(x: str):
        try:
            return int(x)
        except ValueError:
            return x

    try:
        conv_states = {convert_state(s) for s in states}
        conv_accept = {convert_state(s) for s in accept_states}
        conv_start = convert_state(start_state)

        conv_transitions: Dict[Any, Dict[str, Set[Any]]] = {}
        for s, row in transitions.items():
            cs = convert_state(s)
            for a, dsts in row.items():
                conv_transitions.setdefault(cs, {}).setdefault(a, set())
                for t in dsts:
                    ct = convert_state(t)
                    conv_transitions[cs][a].add(ct)
    except Exception as e:
        raise DotParseError(f"Error converting state names: {e}") from e

    nfa = NFA(
        states=conv_states,
        alphabet=alphabet | {EPS},
        start_state=conv_start,
        accept_states=conv_accept,
        transitions=conv_transitions
    )
    return nfa


def epsilon_closure(nfa: NFA, states: Set[Any]) -> Set[Any]:
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for t in nfa.transitions.get(s, {}).get(EPS, set()):
            if t not in closure:
                closure.add(t)
                stack.append(t)
    return closure


def move(nfa: NFA, states: Set[Any], symbol: str) -> Set[Any]:
    res = set()
    for s in states:
        res |= nfa.transitions.get(s, {}).get(symbol, set())
    return res


def nfa_to_dfa(nfa: NFA) -> DFA:
    alphabet = {a for a in nfa.alphabet if a != EPS}

    start_set = frozenset(epsilon_closure(nfa, {nfa.start_state}))
    state_sets: List[FrozenSet[Any]] = [start_set]
    set_to_id: Dict[FrozenSet[Any], int] = {start_set: 0}

    transitions: Dict[int, Dict[str, int]] = {}
    accept_states: Set[int] = set()

    i = 0
    while i < len(state_sets):
        current_set = state_sets[i]
        current_id = i
        transitions[current_id] = {}

        if current_set & nfa.accept_states:
            accept_states.add(current_id)

        for a in alphabet:
            mv = move(nfa, set(current_set), a)
            if not mv:
                continue
            next_set = frozenset(epsilon_closure(nfa, mv))
            if next_set not in set_to_id:
                new_id = len(state_sets)
                state_sets.append(next_set)
                set_to_id[next_set] = new_id
            else:
                new_id = set_to_id[next_set]
            transitions[current_id][a] = new_id

        i += 1

    dfa_states = set(range(len(state_sets)))
    dfa = DFA(
        states=dfa_states,
        alphabet=alphabet,
        start_state=0,
        accept_states=accept_states,
        transitions=transitions,
        state_mapping={i: state_sets[i] for i in range(len(state_sets))}
    )
    return dfa


def make_complete(dfa: DFA) -> None:
    if not dfa.states:
        return
    sink = max(dfa.states) + 1
    dfa.states.add(sink)
    dfa.transitions.setdefault(sink, {})
    for a in dfa.alphabet:
        dfa.transitions[sink][a] = sink
    dfa.state_mapping.setdefault(sink, frozenset())

    for s in list(dfa.states):
        dfa.transitions.setdefault(s, {})
        for a in dfa.alphabet:
            if a not in dfa.transitions[s]:
                dfa.transitions[s][a] = sink


def reachable_subdfa(dfa: DFA) -> DFA:
    stack = [dfa.start_state]
    reachable = set()
    while stack:
        s = stack.pop()
        if s in reachable:
            continue
        reachable.add(s)
        for a in dfa.alphabet:
            if s in dfa.transitions and a in dfa.transitions[s]:
                t = dfa.transitions[s][a]
                if t not in reachable:
                    stack.append(t)

    new_states = reachable
    new_accepts = dfa.accept_states & reachable
    new_transitions: Dict[int, Dict[str, int]] = {}
    for s in new_states:
        row = dfa.transitions.get(s, {})
        new_transitions[s] = {a: t for a, t in row.items() if t in new_states}

    new_mapping: Dict[int, FrozenSet[Any]] = {}
    for s in new_states:
        new_mapping[s] = dfa.state_mapping.get(s, frozenset())

    return DFA(
        states=new_states,
        alphabet=set(dfa.alphabet),
        start_state=dfa.start_state,
        accept_states=new_accepts,
        transitions=new_transitions,
        state_mapping=new_mapping
    )


def minimize_dfa_table(dfa: DFA) -> DFA:
    dfa = copy.deepcopy(dfa)
    make_complete(dfa)
    dfa = reachable_subdfa(dfa)

    states = sorted(dfa.states)
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)

    dist = [[False] * n for _ in range(n)]

    for i in range(n):
        for j in range(i):
            si, sj = states[i], states[j]
            if (si in dfa.accept_states) != (sj in dfa.accept_states):
                dist[i][j] = True

    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i):
                if dist[i][j]:
                    continue
                si, sj = states[i], states[j]
                for a in dfa.alphabet:
                    ti = dfa.transitions[si][a]
                    tj = dfa.transitions[sj][a]
                    ii, jj = idx[ti], idx[tj]
                    if ii < jj:
                        ii, jj = jj, ii
                    if dist[ii][jj]:
                        dist[i][j] = True
                        changed = True
                        break

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i):
            if not dist[i][j]:
                union(i, j)

    rep_to_new: Dict[int, int] = {}
    new_id = 0
    for i in range(n):
        r = find(i)
        if r not in rep_to_new:
            rep_to_new[r] = new_id
            new_id += 1

    old_to_new: Dict[int, int] = {}
    for i, s in enumerate(states):
        r = find(i)
        old_to_new[s] = rep_to_new[r]

    new_states = set(old_to_new.values())
    new_start = old_to_new[dfa.start_state]
    new_accepts = {old_to_new[s] for s in dfa.accept_states}

    new_transitions: Dict[int, Dict[str, int]] = {s: {} for s in new_states}
    for s in states:
        ns = old_to_new[s]
        for a, t in dfa.transitions[s].items():
            nt = old_to_new[t]
            new_transitions[ns][a] = nt

    new_mapping: Dict[int, FrozenSet[Any]] = {}
    for old_s, new_s in old_to_new.items():
        old_set = dfa.state_mapping.get(old_s, frozenset())
        if new_s not in new_mapping:
            new_mapping[new_s] = old_set
        else:
            new_mapping[new_s] = frozenset(set(new_mapping[new_s]) | set(old_set))

    return DFA(
        states=new_states,
        alphabet=set(dfa.alphabet),
        start_state=new_start,
        accept_states=new_accepts,
        transitions=new_transitions,
        state_mapping=new_mapping
    )


def minimize_dfa_hopcroft(dfa: DFA) -> DFA:
    dfa = copy.deepcopy(dfa)
    make_complete(dfa)
    dfa = reachable_subdfa(dfa)

    reachable = dfa.states
    F = dfa.accept_states & reachable
    NF = reachable - F

    P: List[Set[int]] = []
    if F:
        P.append(F)
    if NF:
        P.append(NF)
    W: List[Set[int]] = P.copy()

    while W:
        A = W.pop()
        for c in dfa.alphabet:
            X = {q for q in reachable if dfa.transitions[q][c] in A}
            if not X:
                continue
            newP: List[Set[int]] = []
            for Y in P:
                inter = Y & X
                diff = Y - X
                if inter and diff:
                    newP.append(inter)
                    newP.append(diff)
                    if Y in W:
                        W.remove(Y)
                        W.append(inter)
                        W.append(diff)
                    else:
                        if len(inter) <= len(diff):
                            W.append(inter)
                        else:
                            W.append(diff)
                else:
                    newP.append(Y)
            P = newP

    block_index: Dict[int, int] = {}
    for i, block in enumerate(P):
        for s in block:
            block_index[s] = i

    new_states = set(range(len(P)))
    new_start = block_index[dfa.start_state]
    new_accepts = {block_index[s] for s in F}

    new_transitions: Dict[int, Dict[str, int]] = {i: {} for i in new_states}
    for i, block in enumerate(P):
        rep = next(iter(block))
        for a in dfa.alphabet:
            t = dfa.transitions[rep][a]
            new_transitions[i][a] = block_index[t]

    new_mapping: Dict[int, FrozenSet[Any]] = {}
    for old_s, new_s in block_index.items():
        old_set = dfa.state_mapping.get(old_s, frozenset())
        if new_s not in new_mapping:
            new_mapping[new_s] = old_set
        else:
            new_mapping[new_s] = frozenset(set(new_mapping[new_s]) | set(old_set))

    return DFA(
        states=new_states,
        alphabet=set(dfa.alphabet),
        start_state=new_start,
        accept_states=new_accepts,
        transitions=new_transitions,
        state_mapping=new_mapping
    )


def dfa_to_dot(dfa: DFA, path: str, name: str = "DFA") -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"digraph {name} {{\n")
            f.write("    rankdir=LR;\n")
            f.write("    node [shape=circle];\n")
            f.write("    __start [label=\"\", shape=none];\n")
            f.write(f"    __start -> {dfa.start_state};\n")

            for s in sorted(dfa.states):
                if s in dfa.accept_states:
                    f.write(f"    {s} [shape=doublecircle];\n")

            for s in sorted(dfa.states):
                row = dfa.transitions.get(s, {})
                for a, t in sorted(row.items(), key=lambda x: (x[0], x[1])):
                    f.write(f"    {s} -> {t} [label=\"{a}\"];\n")

            f.write("}\n")
    except OSError as e:
        raise IOError(f"Error writing file '{path}': {e}") from e


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read NFA from DOT, build DFA and minimize it using multiple algorithms."
    )
    parser.add_argument("input_dot", help="Input .dot file describing NFA")
    parser.add_argument(
        "--out-prefix",
        default="automaton",
        help="Prefix for output .dot files. Default: automaton",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    try:
        logger.info("Reading NFA from %s ...", args.input_dot)
        nfa = parse_dot_nfa(args.input_dot)
        logger.info(
            "NFA states: %d, alphabet: %s, start: %s, accepting: %s",
            len(nfa.states),
            sorted(a for a in nfa.alphabet if a != EPS),
            nfa.start_state,
            sorted(nfa.accept_states),
        )
    except DotParseError as e:
        logger.error("%s", e)
        return 1
    except Exception as e:
        logger.error("Unexpected error while parsing NFA: %s", e)
        return 1

    try:
        logger.info("Building DFA (subset construction)...")
        dfa = nfa_to_dfa(nfa)
        logger.info("DFA built: |Q| = %d, |F| = %d", len(dfa.states), len(dfa.accept_states))

        dfa_dot = args.out_prefix + "_dfa.dot"
        dfa_to_dot(dfa, dfa_dot, name="DFA")
        logger.info("DFA saved to %s", dfa_dot)
    except Exception as e:
        logger.error("Error while building/writing DFA: %s", e)
        return 1

    try:
        logger.info("Minimizing DFA (table-filling algorithm)...")
        dfa_min_table = minimize_dfa_table(dfa)
        table_dot = args.out_prefix + "_min_table.dot"
        dfa_to_dot(dfa_min_table, table_dot, name="DFA_MIN_TABLE")
        logger.info(
            "Minimal DFA (table-filling) |Q| = %d saved to %s",
            len(dfa_min_table.states),
            table_dot,
        )
    except Exception as e:
        logger.error("Error while minimizing DFA (table-filling): %s", e)
        return 1

    try:
        logger.info("Minimizing DFA (Hopcroft's algorithm)...")
        dfa_min_h = minimize_dfa_hopcroft(dfa)
        hopcroft_dot = args.out_prefix + "_min_hopcroft.dot"
        dfa_to_dot(dfa_min_h, hopcroft_dot, name="DFA_MIN_HOPCROFT")
        logger.info(
            "Minimal DFA (Hopcroft) |Q| = %d saved to %s",
            len(dfa_min_h.states),
            hopcroft_dot,
        )
    except Exception as e:
        logger.error("Error while minimizing DFA (Hopcroft): %s", e)
        return 1

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
