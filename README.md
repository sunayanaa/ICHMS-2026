# ICHMS-2026
Source code and results related to the paper.

The main, self-contained Python 3 script  runs the complete simulation for all three baseline models (DTSC, Full_Auto, Teleop), aggregates the data, and generates the final plots. 

Requires numpy and matplotlib.

Run the python3 script to produce 2 plots:

1) The aggregated results plot (Bar Charts) corresponding to Section V.A. It compares the three baseline models across the key performance metrics: Time to Task Completion (TTC), Safety Violations (NSV), and Control Interventions (CI).

2) The time-series analysis plot (Line Graph) corresponding to Section V.B. It visualizes the internal logic of the DTSC model, showing the dynamic interplay between Negotiated Trust (T_N), Human Trust (T_H), and Swarm Confidence (C_S) during the GPS Failure and False Positive events.

