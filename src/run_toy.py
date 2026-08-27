"""TOY DML simulation entry. Writes results/tables/toy_dml.csv."""
from __future__ import annotations
import csv, json
from pathlib import Path
import yaml
from events.calendar import load_events
from dml.ate import ols_ate, dml_ate, cate_by_bucket, placebo_ate

def main(root: Path | None = None, cfg_path: str = "configs/toy.yaml") -> dict:
    root = Path(root) if root else Path.cwd()
    cfg = yaml.safe_load((root / cfg_path).read_text()) if (root / cfg_path).exists() else {}
    seed = int(cfg.get("seed", 0))
    true_ate = float(cfg.get("true_ate", 0.05))
    n = int(cfg.get("n", 400))
    panel = load_events(root, seed=seed, true_ate=true_ate, n=n)
    df = panel.frame
    xcols = [c for c in df.columns if c.startswith("x")]
    y = df["car"].to_numpy(); d = df[panel.treatment_col].to_numpy(); x = df[xcols].to_numpy()
    ols = ols_ate(y, d, x)
    dml = dml_ate(y, d, x, seed=seed)
    plc = placebo_ate(y, d, x, seed=seed + 1)
    cate_s = cate_by_bucket(y, d, x, df["size_bucket"].to_numpy())
    cate_v = cate_by_bucket(y, d, x, df["vol_bucket"].to_numpy())
    row = {
        "label": "TOY" if panel.source != "csv" else "EVENTS",
        "source": panel.source, "treatment": panel.treatment_col,
        "true_ate": true_ate, "n": ols["n"],
        "ols_ate": round(ols["ate"], 6), "ols_p": round(ols["p"], 6),
        "dml_ate": round(dml["ate"], 6), "dml_p": round(dml["p"], 6),
        "dml_method": dml["method"],
        "placebo_ate": round(plc["ate"], 6), "placebo_p": round(plc["p"], 6),
        "cate_size": json.dumps(cate_s), "cate_vol": json.dumps(cate_v),
        "note": "TOY simulation; sue_proxy is not I/B/E/S SUE; no WRDS",
    }
    out = root / "results" / "tables" / "toy_dml.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys())); w.writeheader(); w.writerow(row)
    return row

if __name__ == "__main__":
    print(main())
