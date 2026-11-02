import csv
from pathlib import Path

import matplotlib.pyplot as plt

with (Path(__file__).with_name("chart.csv")).open(
    newline="", encoding="utf-8-sig"
) as source:
    rows = list(csv.DictReader(source))

attempts = [int(row["No."]) for row in rows]
static = [int(row["Static"]) for row in rows]
dynamic = [int(row["Dynamic"]) for row in rows]

plt.plot(attempts, static, marker="o", label="Static (historical)")
plt.plot(attempts, dynamic, marker="s", label="Dynamic (historical)")
plt.xlabel("Simulation attempts")
plt.ylabel("Vehicles passed per simulation")
plt.title("Historical repository data (not reproduced by the new benchmark)")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()