from pathlib import Path
import re
import csv
from bs4 import BeautifulSoup

mmd_path = Path("Rayzebio-patent-WO2024073622.mmd")
text = mmd_path.read_text(encoding="utf-8")

out_dir = Path("tables")
out_dir.mkdir(exist_ok=True)

tables_found = 0

# 1) Try HTML-style tables first (DeepSeek often emits <td>)
html_tables = re.findall(r'(<table.*?>.*?</table>)', text, flags=re.S | re.I)

for i, html in enumerate(html_tables, 1):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)

    if rows:
        csv_path = out_dir / f"table_html_{i}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        tables_found += 1

# 2) Fallback: markdown-style tables
md_tables = re.findall(
    r'(\|.+?\|\n\|[-:| ]+\|\n(?:\|.*\|\n)+)',
    text,
    flags=re.S
)

for i, table in enumerate(md_tables, 1):
    lines = [l.strip().strip("|") for l in table.strip().splitlines()]
    rows = [[c.strip() for c in line.split("|")] for line in lines]

    csv_path = out_dir / f"table_md_{i}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    tables_found += 1

print(f"Extracted {tables_found} tables into {out_dir}/")
