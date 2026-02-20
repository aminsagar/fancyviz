import re
import csv
from pathlib import Path

md = Path("amin.mmd").read_text()

tables = re.findall(r'(\|.+?\|\n\|[-:| ]+\|\n(?:\|.*\|\n)+)', md, flags=re.S)

out_dir = Path("tables")
out_dir.mkdir(exist_ok=True)

for i, table in enumerate(tables):
    lines = [l.strip().strip("|") for l in table.strip().splitlines()]
    rows = [[c.strip() for c in line.split("|")] for line in lines]
    
    csv_path = out_dir / f"table_{i+1}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

print(f"Saved {len(tables)} tables to tables/")
