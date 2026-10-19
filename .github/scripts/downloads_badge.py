#!/usr/bin/env python3
"""Combine PyPI (pypistats, incl. mirrors) and Bioconda download counts
into a shields.io endpoint JSON. Stdlib only.

Usage: downloads_badge.py <package> <data_dir>
Reads/writes <data_dir>/history.json and writes <data_dir>/downloads.json.
"""
import datetime
import json
import os
import sys
import time
import urllib.request

UA = {"User-Agent": "downloads-badge (github actions)"}


def get_json(url, tries=3):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (i + 1))
    raise last


def human(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def main():
    pkg, data_dir = sys.argv[1], sys.argv[2]
    hist_path = os.path.join(data_dir, "history.json")
    badge_path = os.path.join(data_dir, "downloads.json")

    hist = {}
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            hist = json.load(f)
    daily = hist.get("pypi_daily", {})
    prev_pypi = int(hist.get("pypi_total", 0))
    prev_conda = int(hist.get("conda_total", 0))

    # PyPI daily downloads (with mirrors), merged into persistent history
    try:
        data = get_json(f"https://pypistats.org/api/packages/{pkg}/overall?mirrors=true")
        for row in data.get("data", []):
            if row.get("category") != "with_mirrors":
                continue
            d, n = row["date"], int(row["downloads"])
            daily[d] = max(n, int(daily.get(d, 0)))
    except Exception as e:  # noqa: BLE001
        print(f"WARN: pypistats failed: {e}", file=sys.stderr)
    pypi_total = max(sum(daily.values()), prev_pypi)

    # Bioconda cumulative downloads
    try:
        data = get_json(f"https://api.anaconda.org/package/bioconda/{pkg}")
        conda_total = sum(int(f.get("ndownloads", 0)) for f in data.get("files", []))
    except Exception as e:  # noqa: BLE001
        print(f"WARN: anaconda failed: {e}", file=sys.stderr)
        conda_total = prev_conda
    conda_total = max(conda_total, prev_conda)

    total = pypi_total + conda_total
    hist = {
        "package": pkg,
        "pypi_total": pypi_total,
        "conda_total": conda_total,
        "total": total,
        "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "pypi_daily": dict(sorted(daily.items())),
    }
    os.makedirs(data_dir, exist_ok=True)
    with open(hist_path, "w") as f:
        json.dump(hist, f, indent=1)
        f.write("\n")
    badge = {
        "schemaVersion": 1,
        "label": "total downloads",
        "message": human(total),
        "color": "brightgreen",
    }
    with open(badge_path, "w") as f:
        json.dump(badge, f)
        f.write("\n")
    print(f"{pkg}: pypi={pypi_total} conda={conda_total} total={total} -> {badge['message']}")


if __name__ == "__main__":
    main()
