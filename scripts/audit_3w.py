"""Описательный аудит реальных экземпляров Petrobras 3W (docs/protocol_3w_audit.md).

    python -m scripts.audit_3w            # → results/threew_audit_v1/

Нужна загрузка scripts/download_3w.py (data/raw/3w/). Без моделей и без выбора эталона.
summary.md перезаписывается; выводы — в docs/results/threew_audit.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow

from scripts.download_3w import ROOT, check
from src import threew_audit as ta

OUT = Path("results/threew_audit_v1")
PAPER_COUNTS = {"0": 594, "1": 4, "2": 22, "3": 32, "4": 343, "5": 11, "6": 6, "7": 36, "8": 14, "9": 57}
PAPER_WELLS = 42


def segments(ev: ta.Event, n: int) -> dict[str, slice]:
    out = {"whole": slice(0, n)}
    if ev.tau >= 0:
        out["old"] = slice(ev.tau - ev.old_len, ev.tau)
        out["transient"] = slice(ev.tau, ev.tau + ev.trans_len)
        if ev.new_len:
            out["new"] = slice(ev.new_start, ev.new_start + ev.new_len)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "manifest.json").read_text())
    check(manifest)

    inv, per, var, elig = [], [], [], []
    for e in manifest["files"]:
        f = ROOT / e["local"]
        d = pd.read_parquet(f)
        name, cls_dir = f.name, e["class_dir"]
        well = name.split("_")[0]
        lab = d["class"].to_numpy(dtype=float)
        n = len(d)
        ev = ta.event_of(lab)
        tc = ta.time_check(d.index)
        ps = ta.periods(lab)
        per += [{"file": name, "class_dir": cls_dir, "label": l, "start": s, "length": ln} for l, s, ln in ps]
        state = d["state"].to_numpy(dtype=float) if "state" in d else np.full(n, np.nan)
        segs = segments(ev, n)
        row = {"file": name, "class_dir": cls_dir, "well": well, **tc, "structure": ev.structure, "tau": ev.tau,
               "old_len": ev.old_len, "trans_len": ev.trans_len, "new_len": ev.new_len,
               "from_tau": ev.total_from_tau, "unlabeled": int(np.isnan(lab).sum()),
               "labels": " ".join(sorted({str(int(x)) for x in lab[np.isfinite(lab)]}))}
        for sname in ("old", "new"):
            if sname in segs:
                st = state[segs[sname]]
                row[f"state_{sname}"] = " ".join(sorted({str(int(x)) for x in st[np.isfinite(st)]})) or "NaN"
        numeric = [c for c in d.columns if c not in ("class", "state")]
        both = 0
        for c in numeric:
            vals = d[c].to_numpy(dtype=float)
            flags = {}
            for sname, sl in segs.items():
                v = vals[sl]
                flags[sname] = ta.usable(v)
                var.append({"file": name, "class_dir": cls_dir, "variable": c, "segment": sname, "n": len(v),
                            "usable": flags[sname], "nan_share": float(np.isnan(v).mean()) if len(v) else np.nan,
                            "frozen_share": ta.frozen_share(v),
                            "interp_share": ta.interp_share(v) if sname in ("old", "whole") else np.nan})
            if not c.startswith("ESTADO") and flags.get("old") and flags.get("new"):
                both += 1
        row["usable_vars_whole"] = sum(1 for c in numeric if not c.startswith("ESTADO")
                                       and ta.usable(d[c].to_numpy(dtype=float)))
        row["usable_vars_old_and_new"] = both
        inv.append(row)
        for k in ta.SCALES:
            for L, H in ta.CONFIGS:
                if ev.tau >= 0:
                    s = ta.sufficient(ev, k, L, H)
                    elig.append({"file": name, "class_dir": cls_dir, "well": well, "k": k, "L": L, "H": H, **s,
                                 "all_with_var": s["all"] and both > 0})
                elif ev.structure == "0 only":
                    zero = max((ln for l, _, ln in ps if l == 0), default=0)
                    elig.append({"file": name, "class_dir": cls_dir, "well": well, "k": k, "L": L, "H": H,
                                 "control_longest0_blocks": ta.blocks(zero, k),
                                 "control_ok": ta.blocks(zero, k) >= L + H,
                                 "control_ok_with_var": ta.blocks(zero, k) >= L + H and row["usable_vars_whole"] > 0})

    inv, per, var, elig = map(pd.DataFrame, (inv, per, var, elig))
    inv.to_csv(OUT / "inventory.csv", index=False)
    per.to_csv(OUT / "periods.csv", index=False)
    var.to_csv(OUT / "variables.csv.gz", index=False)
    elig.to_csv(OUT / "eligibility.csv", index=False)

    # Сводки
    by_class = inv.groupby("class_dir").agg(files=("file", "size"), wells=("well", "nunique"),
                                            median_n=("n", "median"), median_old=("old_len", "median"),
                                            median_trans=("trans_len", "median"),
                                            median_new=("new_len", "median"))
    by_class["paper"] = pd.Series(PAPER_COUNTS)
    structure = pd.crosstab(inv.class_dir, inv.structure)
    ev_rows = elig[elig["all"].notna()]  # строки событий (у контроля столбца all нет)
    suff = ev_rows.groupby(["k", "L", "H"]).agg(events=("file", "size"), old=("old", "sum"), span=("span", "sum"),
                                                 new=("new", "sum"), all=("all", "sum"),
                                                 all_with_var=("all_with_var", "sum")).reset_index()
    wells_ok = ev_rows[ev_rows.all_with_var.astype(bool)].groupby(["k", "L", "H"]).well.nunique()
    suff = suff.merge(wells_ok.rename("wells_all_with_var").reset_index(), how="left", on=["k", "L", "H"]).fillna(0)
    suff_cls = ev_rows.groupby(["k", "L", "H", "class_dir"])["all_with_var"].sum().unstack(fill_value=0)
    ctrl = elig[elig["control_ok"].notna()].copy() if "control_ok" in elig else pd.DataFrame()
    if len(ctrl):
        ctrl["well_ok"] = ctrl.well.where(ctrl.control_ok_with_var.astype(bool))
        ctrl_sum = ctrl.groupby(["k", "L", "H"]).agg(instances=("file", "size"), ok=("control_ok", "sum"),
                                                     ok_with_var=("control_ok_with_var", "sum"),
                                                     wells_ok=("well_ok", "nunique")).reset_index()
    vv = var[(var.segment.isin(["old", "new", "whole"])) & ~var.variable.str.startswith("ESTADO")]
    var_sum = vv.groupby(["variable", "segment"]).agg(usable_share=("usable", "mean"),
                                                      median_frozen=("frozen_share", "median"),
                                                      median_interp=("interp_share", "median")).unstack()
    # Пересечения файлов одной скважины во времени
    inv["start_ts"], inv["end_ts"] = pd.to_datetime(inv.start), pd.to_datetime(inv.end)
    ov = []
    for well, g in inv.sort_values("start_ts").groupby("well"):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            gap = (g.start_ts[i] - g.end_ts[i - 1]).total_seconds()
            ov.append({"well": well, "prev": g.file[i - 1], "next": g.file[i], "gap_s": gap})
    ov = pd.DataFrame(ov)
    ov.to_csv(OUT / "well_gaps.csv", index=False)
    n_overlap = int((ov.gap_s < 0).sum()) if len(ov) else 0
    n_adjacent = int(((ov.gap_s >= 0) & (ov.gap_s <= 1)).sum()) if len(ov) else 0

    # Графики: по одному экземпляру на класс — первый по имени файла (для классов с переходом — с переходом).
    figs = []
    for cls, g in inv.sort_values("file").groupby("class_dir"):
        cand = g[g.structure.str.contains("T")] if (g.structure.str.contains("T")).any() else g
        r = cand.iloc[0]
        d = pd.read_parquet(ROOT / "files" / cls / r.file)
        cols = [c for c in d.columns if c not in ("class", "state") and not c.startswith("ESTADO")
                and ta.usable(d[c].to_numpy(dtype=float))][:4]
        fig, axes = plt.subplots(len(cols) + 1, 1, figsize=(10, 1.8 * (len(cols) + 1)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, c in zip(axes, cols):
            ax.plot(np.arange(len(d)), d[c].to_numpy(dtype=float), lw=0.6)
            ax.set_ylabel(c, fontsize=7)
        axes[-1].plot(np.arange(len(d)), d["class"].to_numpy(dtype=float), lw=0.8, color="k")
        axes[-1].set_ylabel("class", fontsize=7)
        axes[-1].set_xlabel("секунды от начала файла")
        fig.suptitle(f"класс {cls}: {r.file} ({r.structure})", fontsize=9)
        fig.tight_layout()
        path = OUT / f"fig_class_{cls}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        figs.append(path.name)

    wells = inv.well.nunique()
    val = {"manifest_check": "passed", "n_files": len(inv), "expected_files": 1119,
           "counts_match_paper": {k: int(by_class.files.get(k, 0)) == v for k, v in PAPER_COUNTS.items()},
           "wells": int(wells), "paper_wells": PAPER_WELLS,
           "all_monotonic": bool(inv.monotonic.all()), "files_with_duplicates": int((inv.duplicates > 0).sum()),
           "files_all_steps_1s": int((inv.share_step_1s == 1).sum()),
           "time_overlaps_same_well": n_overlap, "adjacent_files_same_well": n_adjacent}
    val["passed"] = bool(val["n_files"] == val["expected_files"] and val["all_monotonic"])
    (OUT / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=2))
    meta = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "protocol": "docs/protocol_3w_audit.md",
            "commit": manifest["commit"], "scales": list(ta.SCALES), "configs": [list(c) for c in ta.CONFIGS],
            "seconds": round(time.perf_counter() - t0, 1), "figures": figs,
            "env": f"python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, "
                   f"pyarrow {pyarrow.__version__}"}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    def md(df):
        df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
        cols = [str(c) if not isinstance(c, tuple) else "/".join(map(str, c)) for c in df.columns]
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for row in df.itertuples(index=False):
            lines.append("| " + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in row) + " |")
        return "\n".join(lines)

    text = ["# Аудит Petrobras 3W (реальные экземпляры)", "",
            "Создано `scripts/audit_3w.py` по `docs/protocol_3w_audit.md`. Перезаписывается; выводы — "
            "`docs/results/threew_audit.md`.", "", "## Проверки", "", "```", json.dumps(val, ensure_ascii=False, indent=1),
            "```", "", "## По классам", "", md(by_class), "", "## Структура экземпляров", "", md(structure), "",
            "## Достаточность длин событий (протокол v1)", "", md(suff), "", "### По классам (all_with_var)", "",
            md(suff_cls), "", "## Стабильный контроль (класс 0, самый длинный период 0 ≥ L + H)", "",
            md(ctrl_sum) if len(ctrl) else "нет", "", "## Переменные (без ESTADO)", "", md(var_sum.round(3)), ""]
    (OUT / "summary.md").write_text("\n".join(text))
    print(json.dumps(val, ensure_ascii=False))
    print(suff.to_string(index=False))
    print(f"Готово за {meta['seconds']} с")


if __name__ == "__main__":
    main()
