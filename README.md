# nfa2dfa

A small tool for working with finite automata:

- reads an **NFA** from a Graphviz `.dot` file;
- builds an equivalent **DFA** (subset construction);
- minimizes the DFA using **two different algorithms**:
  - the table-filling (pairwise) minimization algorithm,
  - **Hopcroft's algorithm**;
- generates `.dot` files for all resulting automata so they can be visualized with Graphviz.

## Requirements

- Python 3.10+ (it will also work with 3.8/3.9 if you remove the `str | None` annotation, but the default target is 3.10+);
- no external libraries — only the Python standard library is used.

## Input NFA format (.dot)

The program expects a `.dot` file roughly of this form:

```dot
digraph NFA {
    rankdir=LR;
    node [shape=circle];

    __start [label="", shape=none];
    __start -> 0;

    21 [shape=doublecircle];

    0 -> 1 [label="a"];
    0 -> 2 [label="b"];
    0 -> 3 [label="c"];
    0 -> 4 [label="ε"];
    1 -> 0 [label="a"];
    2 -> 0 [label="b"];
    3 -> 0 [label="c"];
    ...
}
```

Rules:

- The start state is specified via a special node `__start`:

  ```dot
  __start [label="", shape=none];
  __start -> 0;
  ```

  Here `0` is the start state.

- Accepting states are marked with `shape=doublecircle`:

  ```dot
  21 [shape=doublecircle];
  ```

- Transitions are specified as:

  ```dot
  0 -> 1 [label="a"];
  19 -> 20 [label="a,b,c"];
  20 -> 21 [label="b,ε"];
  ```

  - If there are several symbols on an edge, they are separated by commas.
  - Epsilon transitions use the symbol `ε` (Greek epsilon).
    If the label is `b,ε`, this means a transition on `b` **and** an epsilon transition.

- State names should be “simple” (like Graphviz defaults):
  integers (`0`, `1`, …) or identifiers made of `\w` (Latin letters/digits/underscore).  
  The node `__start` is reserved for the start arrow and **is not** considered an NFA state.

## What the program does

1. **Parsing the NFA** from `.dot`:
   - extracts the set of states;
   - finds the start and accepting states;
   - builds the transition table `transitions[state][symbol] = {next_states}`;
   - extracts the alphabet (all symbols on edges except `ε`).

2. **Building a DFA**:
   - uses the standard subset construction algorithm;
   - DFA states are subsets of NFA states;
   - in code, DFA states are numbered `0..n-1`;
   - field `state_mapping` stores the mapping  
     `dfa_state_number -> set of NFA states`.

3. **DFA minimization (2 algorithms)**:
   - **Table-filling algorithm (pairwise / table-filling)**:
     - builds a table of distinguishable pairs of states;
     - marks pairs `(p, q)` where exactly one of them is accepting;
     - iteratively propagates “distinguishable” marks;
     - unmarked pairs form equivalence classes.

   - **Hopcroft's algorithm**:
     - initial partition: `F` (accepting states) and `Q \ F`;
     - then iteratively refines the partition using reverse transitions;
     - the result is a minimal DFA with time complexity `O(|Σ| * |Q| log |Q|)`.

   In both algorithms:
   - the DFA is first made **complete** (a sink state is added);
   - missing transitions are redirected to the sink;
   - then only **reachable** states are kept.

4. **DOT generation**:
   - for three automata:
     - the original DFA (from the NFA),
     - the minimal DFA (table-filling algorithm),
     - the minimal DFA (Hopcroft's algorithm);
   - output format:

     ```dot
     digraph DFA_MIN_TABLE {
         rankdir=LR;
         node [shape=circle];
         __start [label="", shape=none];
         __start -> 0;
         0 [shape=circle];
         1 [shape=doublecircle];
         ...
         0 -> 1 [label="a"];
         ...
     }
     ```

## Usage

```bash
python automata_minimizer.py input.dot --out-prefix result
```

Where:

- `input.dot` is the NFA file in the format described above;
- `--out-prefix result` is the prefix for output file names (by default `automaton`).

After running you will get:

- `result_dfa.dot` — DFA constructed from the given NFA;
- `result_min_table.dot` — minimal DFA (table-filling algorithm);
- `result_min_hopcroft.dot` — minimal DFA (Hopcroft's algorithm).

## Visualization

To visualize the automata, use Graphviz. For example:

```bash
dot -Tpng result_dfa.dot -o dfa.png
dot -Tpng result_min_table.dot -o dfa_min_table.png
dot -Tpng result_min_hopcroft.dot -o dfa_min_hopcroft.png
```

## Notes

- Both minimization algorithms (table-filling and Hopcroft) produce **automata with the same number of states**.  
  The only difference may be the numbering of states.
- The sink state will appear in the minimal DFA if it is required to make the automaton complete.
- Internally, DFA states are always numbered with integers `0..n-1`,  
  but you can use `state_mapping` to see which sets of NFA states they correspond to.
