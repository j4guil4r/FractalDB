import os, csv

base = "datos/image_demo"   # en tu host
rows = []
i = 1

for fname in sorted(os.listdir(base)):
    if not fname.lower().endswith((".wav", ".ogg", ".flac", ".mp3", ".bmp")):
        continue
    label = fname.split("-")[1] if "-" in fname else "unknown"
    container_path = f"/datos/image_demo/{fname}"
    rows.append((i, label, container_path))
    i += 1

with open("image_demo.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "label", "image_path"])
    w.writerows(rows)
