# ICHMS-2026
# Trustworthy HSI Simulation

This repository contains the simulation code for the paper:
"Dynamic Trust Negotiation for Safe Human-Swarm Interaction: A Decentralized Control Approach"

This code is provided for independent verification of the results presented in Section V.

Files
`main_simulation.py`: The main, self-contained Python 3 script. This file runs the complete simulation for all three baseline models (DTSC, Full_Auto, Teleop), aggregates the data, and generates the final plots.
`results_baseline_comparison.png`: The aggregated results plot (Bar Charts) corresponding to Section V.A.
`results_trust_dynamics.png`: The time-series analysis plot (Line Graph) corresponding to Section V.B.

How to Run
1.  This script is best run in a Google Colab notebook or any Python 3 environment with `numpy` and `matplotlib` installed.
2.  Paste the contents of `main_simulation.py` into a cell and run it.
3.  The script will execute all simulation runs (50 per baseline) and then generate the two results plots.
