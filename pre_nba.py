"""Pre-NBA signals model: the trained pipeline's insights, minus the NBA sample.

Same 398-player universe and breakout labels as the main model, but trained
ONLY on features that exist before a player logs an NBA minute: college
translation, G-League production, combine, age, draft pedigree. Two uses:

  python pre_nba.py                  # archetype signal table: which pre-NBA
                                     # stats drive NBA breakouts, by archetype
  python pre_nba.py "jaden akins"    # signal profile for a player with no NBA
                                     # minutes (G-League-only, draft-and-stash)

This is NOT a prospect grade. For G-League-only players the historical base
rate of jumping straight to an NBA breakout is 0.2% — the output is a signal
profile ("what stands out analytically"), not a probability of making it.
"""
import sys

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score

import leap_model as lm
from college_translation import college_features, college_table, norm_name

# every main-model feature that does NOT come from the player's NBA minutes
NBA_SAMPLE = {"MIN12", "GP12", "MPG12", "PTS36", "REB36", "AST36", "STL36",
              "BLK36", "TOV36", "FGA36", "FG3A36", "FTA36", "OREB36", "STOCKS36",
              "TS_SHRUNK", "FG3PCT_SHRUNK", "FT_PCT_SHRUNK"}
PRE_FEATURES = [f for f in lm.FEATURES if f not in NBA_SAMPLE]


def train(verbose=True):
    """Fit on the labeled universe (cached by player_report --rebuild)."""
    allp = pd.read_csv(lm.DATA / "scored_universe.csv")
    hist = allp[allp["POOL"] == "historical"]
    tr = hist[hist["COHORT"] < lm.HOLDOUT_FROM]
    ho = hist[hist["COHORT"] >= lm.HOLDOUT_FROM]

    def fit(d):
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, eval_metric="auc",
            scale_pos_weight=(d["BREAKOUT"] == 0).sum() / d["BREAKOUT"].sum())
        m.fit(d[PRE_FEATURES], d["BREAKOUT"])
        return m

    auc = roc_auc_score(ho["BREAKOUT"], fit(tr).predict_proba(ho[PRE_FEATURES])[:, 1])
    if verbose:
        print(f"pre-NBA-signals model — holdout AUC {auc:.3f} "
              f"(main model 0.782, coin flip 0.500)\n")
    return fit(hist), hist


def archetype_signals(model, hist, top=6):
    """Which pre-NBA stats matter per archetype, by mean |SHAP|, with direction."""
    sv = pd.DataFrame(shap.TreeExplainer(model).shap_values(hist[PRE_FEATURES]),
                      columns=PRE_FEATURES, index=hist.index)
    print("=== WHAT PRE-NBA SIGNALS DRIVE BREAKOUT, BY ARCHETYPE ===")
    rows = []
    for arch, g in hist.groupby("ARCHETYPE"):
        strength = sv.loc[g.index].abs().mean().sort_values(ascending=False)
        print(f"\n{arch} (n={len(g)}, {g['BREAKOUT'].mean():.0%} broke out)")
        for f in strength.head(top).index:
            direction = g[f].corr(sv.loc[g.index, f])
            arrow = "higher = better" if direction > 0 else "lower = better"
            print(f"  {f:28s} weight {strength[f]:.3f}  ({arrow})")
            rows.append({"ARCHETYPE": arch, "FEATURE": f,
                         "WEIGHT": strength[f], "DIRECTION": arrow})
    pd.DataFrame(rows).to_csv(lm.DATA / "archetype_signals.csv", index=False)
    print("\n-> data/archetype_signals.csv")


def build_row(name):
    """Assemble a pre-NBA feature row for a player with no NBA minutes,
    from the G-League / college / combine / draft caches."""
    key = norm_name(name)
    row, cohort, pid, display = {}, None, None, None

    glg = pd.read_csv(lm.DATA / "gleague_totals.csv")
    g = glg[glg["PLAYER_NAME"].map(norm_name) == key]
    if len(g):
        pid = g["PLAYER_ID"].iloc[0]
        display = g["PLAYER_NAME"].iloc[0]
        cohort = int(g["SEASON"].str[:4].astype(int).min())
        mini = pd.DataFrame({"PLAYER_ID": [pid], "COHORT": [cohort]})
        feats = lm.gleague_features(mini, glg).iloc[0]
        row.update({c: feats[c] for c in feats.index if c.startswith(("GLG_", "HAS_GLG"))})
        row["AGE"] = g.sort_values("SEASON")["AGE"].iloc[0]

    ct = college_table()
    if (ct["key"] == key).any():
        display = display or name.title()
        cohort = cohort or 2026  # college-only: treat as current
        cf = college_features(pd.DataFrame({"PLAYER_NAME": [display], "COHORT": [cohort]})).iloc[0]
        row.update(cf.to_dict())
        row.setdefault("AGE", cf["COLLEGE_AGE"] + (cohort - 2026) + 1
                       if pd.notna(cf["COLLEGE_AGE"]) else np.nan)

    if display is None:
        return None, None, None

    draft = pd.read_csv(lm.DATA / "draft_history.csv")
    hit = draft[draft["PLAYER_NAME"].map(norm_name) == key] if pid is None else \
        draft[draft["PERSON_ID"] == pid]
    row["PICK"] = hit["OVERALL_PICK"].iloc[0] if len(hit) else 75
    row["UNDRAFTED"] = int(not len(hit))

    if pid is not None:
        comb = pd.read_csv(lm.DATA / "combine.csv")
        c = comb[comb["PLAYER_ID"] == pid]
        for f in ["HEIGHT_WO_SHOES", "WINGSPAN", "WEIGHT", "STANDING_VERTICAL_LEAP",
                  "MAX_VERTICAL_LEAP", "LANE_AGILITY_TIME", "THREE_QUARTER_SPRINT"]:
            row[f] = pd.to_numeric(c[f], errors="coerce").iloc[-1] if len(c) else np.nan
    ws, ht = row.get("WINGSPAN", np.nan), row.get("HEIGHT_WO_SHOES", np.nan)
    row["WINGSPAN_RATIO"] = ws / ht if pd.notna(ws) and pd.notna(ht) else np.nan

    full = pd.Series({f: row.get(f, np.nan) for f in PRE_FEATURES}, name=display)
    return full, display, cohort


def crude_archetype(row, key):
    ht = row.get("HEIGHT_WO_SHOES")
    if pd.isna(ht):
        ct = college_table()
        hit = ct[ct["key"] == key]
        ht = hit["height_in"].iloc[-1] if len(hit) else np.nan
    if pd.isna(ht):
        return "wing (unknown size)"
    return "big" if ht >= 80 else "guard" if ht <= 75 else "wing"


def report(name):
    row, display, cohort = build_row(name)
    if row is None:
        print(f'"{name}" not found in G-League or college data either.')
        return
    model, hist = train(verbose=False)
    prob = model.predict_proba(row.to_frame().T[PRE_FEATURES])[0, 1]
    ref = model.predict_proba(hist[PRE_FEATURES])[:, 1]
    sv = shap.TreeExplainer(model).shap_values(row.to_frame().T[PRE_FEATURES])[0]
    s = pd.Series(sv, index=PRE_FEATURES)

    arch = crude_archetype(row, norm_name(name))
    print(f"=== PRE-NBA SIGNAL PROFILE: {display} ===")
    print(f"first pro season {cohort} | {arch} | no NBA minutes — this is a signal")
    print("profile from college/G-League/combine data only, NOT a breakout prediction.")
    print(f"(G-League-only players historically jump to an NBA breakout 0.2% of the time)\n")
    print(f"pre-NBA signal score: {prob:.2f} — stronger than {(ref < prob).mean():.0%} "
          f"of the historical universe's pre-NBA profiles")
    glmin = row.get("GLG_MIN12")
    print(f"evidence: {0 if pd.isna(glmin) else glmin:.0f} G-League min, "
          f"college data: {'yes' if row.get('HAS_COLLEGE') == 1 else 'no'}, "
          f"combine: {'yes' if pd.notna(row.get('HEIGHT_WO_SHOES')) else 'no'}\n")

    order = s.abs().sort_values(ascending=False).index
    hist_ref = hist[PRE_FEATURES]

    def line(f):
        pct = (hist_ref[f] < row[f]).mean() * 100
        v = f"{row[f]:.2f}" if abs(row[f]) < 10 else f"{row[f]:.0f}"
        return f"  {f:28s} {v:>7s}  (p{pct:.0f} of universe)"

    pos = [f for f in order if s[f] > 0 and pd.notna(row[f])][:4]
    neg = [f for f in order if s[f] < 0 and pd.notna(row[f])][:4]
    print("ANALYTICAL STRENGTHS (vs players who later broke out)")
    print("\n".join(line(f) for f in pos) or "  none of note")
    print("\nANALYTICAL WEAKNESSES")
    print("\n".join(line(f) for f in neg) or "  none of note")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        report(" ".join(sys.argv[1:]))
    else:
        model, hist = train()
        archetype_signals(model, hist)
