# Nash Equilibrium Calculator

This repository provides a Python-based toolkit for computing Nash equilibria in normal-form and small extensive-form games. It supports mixed-strategy equilibrium computation and can be used for educational purposes, research prototypes, or integration into strategic simulation systems.

## Features

- Supports 2-player normal-form games  
- Mixed strategy equilibrium solver (linear programming approach)  
- Payoff matrix input via text or programmatic interface  
- Optional support for multiple equilibria  
- Easily extensible to compute best responses and dominance relations

## Repository Structure

```
Nash-Equilibrium-Calculator/
├── nash_solver.py         # Core functions for computing Nash equilibria
├── example_games.py       # Sample games (Prisoner's Dilemma, Matching Pennies, etc.)
├── input_parser.py        # Optional parser for text-based game input
├── notebooks/
│   └── demo_equilibria.ipynb  # Interactive notebook demo
└── README.md
```

## Quick Start

Clone the repo and run the example:

```bash
git clone https://github.com/YushenSun/Nash-Equilibrium-Calculator.git
cd Nash-Equilibrium-Calculator
python example_games.py
```

Alternatively, open the Jupyter notebook for a step-by-step interactive demo:

```bash
jupyter notebook notebooks/demo_equilibria.ipynb
```

## Example

```python
from nash_solver import compute_nash_equilibrium

A = [[3, 1], [0, 2]]  # Player A's payoffs
B = [[2, 0], [3, 1]]  # Player B's payoffs

equilibria = compute_nash_equilibrium(A, B)
print(equilibria)
```

## Applications

- Game theory education and experimentation  
- AI decision-making and strategy modeling  
- Testing equilibrium sensitivity and comparative statics

## To Do

- Add support for n-player games  
- Visualize payoff spaces and best-response diagrams  
- Extend to sequential games with backward induction

## License

MIT
