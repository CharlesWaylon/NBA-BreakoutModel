"""Player lookup: breakout score + analytical strengths/weaknesses report.

    python player_report.py "isaiah joe"
    python player_report.py --rebuild        # refresh the cache after new data

First run builds a cache (data/scored_universe.csv + data/shap_values.csv);
lookups after that are instant. Covers the 328 labeled historical players and
all current-board players (end-of-bench + fallen angels).
"""
import sys
from difflib import get_close_matches

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import leap_model as lm
import score_current as sc

UNIVERSE = lm.DATA / "scored_universe.csv"
SHAPVALS = lm.DATA / "shap_values.csv"

LABELS = {
    "TS_SHRUNK": "true shooting % (shrunk)", "FT_PCT_SHRUNK": "FT% — shooting-touch proxy",
    "FG3PCT_SHRUNK": "3P% (shrunk)", "FG3A36": "3PA per 36", "PTS36": "points per 36",
    "REB36": "rebounds per 36", "AST36": "assists per 36", "STL36": "steals per 36",
    "BLK36": "blocks per 36", "TOV36": "turnovers per 36", "OREB36": "off. rebounds per 36",
    "STOCKS36": "steals+blocks per 36", "FGA36": "FGA per 36", "FTA36": "FTA per 36",
    "MIN12": "NBA minutes, yrs 1-2", "GP12": "NBA games, yrs 1-2", "MPG12": "MPG, yrs 1-2",
    "AGE": "age at debut", "PICK": "draft pick", "UNDRAFTED": "undrafted",
    "GLG_PTS36": "G-League pts/36", "GLG_REB36": "G-League reb/36",
    "GLG_AST36": "G-League ast/36", "GLG_STOCKS36": "G-League stl+blk/36",
    "GLG_TOV36": "G-League tov/36", "GLG_TS": "G-League true shooting",
    "GLG_MIN12": "G-League minutes", "HAS_GLG": "G-League stint",
    "PROJ_PTS36": "college-projected pts/36", "PROJ_REB36": "college-projected reb/36",
    "PROJ_AST36": "college-projected ast/36", "PROJ_STL36": "college-projected stl/36",
    "PROJ_BLK36": "college-projected blk/36", "PROJ_TOV36": "college-projected tov/36",
    "PROJ_TS": "college-projected TS%", "COLLEGE_BPM": "college BPM",
    "COLLEGE_USG": "college usage", "COLLEGE_AGE": "college age", "HAS_COLLEGE": "college data",
    "HEIGHT_WO_SHOES": "height", "WEIGHT": "weight", "WINGSPAN": "wingspan",
    "WINGSPAN_RATIO": "wingspan/height", "STANDING_VERTICAL_LEAP": "standing vertical",
    "MAX_VERTICAL_LEAP": "max vertical", "LANE_AGILITY_TIME": "lane agility (s)",
    "THREE_QUARTER_SPRINT": "3/4-court sprint (s)",
}


def build_cache():
    tr, cur = sc.build()
    tr["POOL"] = "historical"
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, eval_metric="auc",
        scale_pos_weight=(tr["BREAKOUT"] == 0).sum() / tr["BREAKOUT"].sum())
    model.fit(tr[lm.FEATURES], tr["BREAKOUT"])

    allp = pd.concat([tr, cur], ignore_index=True)
    allp["PROB"] = model.predict_proba(allp[lm.FEATURES])[:, 1]
    allp["PEDIGREE"] = allp["PICK"].map(lm.pedigree)

    # comps for everyone, within archetype, against labeled history (skip self)
    med = tr[lm.COMP_FEATURES].median()
    allp["COMPS"], allp["COMP_BREAKOUT_RATE"] = "", np.nan
    for arch, g in allp.groupby("ARCHETYPE"):
        pool = tr[tr["ARCHETYPE"] == arch]
        scaler = StandardScaler()
        knn = NearestNeighbors(n_neighbors=min(9, len(pool))).fit(
            scaler.fit_transform(pool[lm.COMP_FEATURES].fillna(med)))
        _, idx = knn.kneighbors(scaler.transform(g[lm.COMP_FEATURES].fillna(med)))
        for i, nn in zip(g.index, idx):
            sub = pool.iloc[nn]
            sub = sub[sub["PLAYER_ID"] != allp.loc[i, "PLAYER_ID"]].head(8)
            allp.loc[i, "COMPS"] = ", ".join(sub["PLAYER_NAME"])
            allp.loc[i, "COMP_BREAKOUT_RATE"] = sub["BREAKOUT"].mean()

    sv = shap.TreeExplainer(model).shap_values(allp[lm.FEATURES])
    allp.to_csv(UNIVERSE, index=False)
    pd.DataFrame(sv, columns=lm.FEATURES).to_csv(SHAPVALS, index=False)
    print(f"cache built: {len(allp)} players")


def confidence(r):
    mins = r["MIN12"] + (0 if pd.isna(r.get("GLG_MIN12")) else r["GLG_MIN12"])
    tier = "HIGH" if mins >= 1200 else "MEDIUM" if mins >= 500 else "LOW"
    parts = [f"{r['MIN12']:.0f} NBA min",
             f"{0 if pd.isna(r.get('GLG_MIN12')) else r['GLG_MIN12']:.0f} G-League min",
             f"college data: {'yes' if r.get('HAS_COLLEGE') == 1 else 'no'}",
             f"combine: {'yes' if pd.notna(r.get('HEIGHT_WO_SHOES')) else 'no'}"]
    return tier, ", ".join(parts)


def explain_exclusion(hit):
    """Player exists in NBA data but isn't in the scored universe — say why."""
    hit = hit[hit["PLAYER_NAME"] == hit["PLAYER_NAME"].iloc[0]].copy()
    name = hit["PLAYER_NAME"].iloc[0]
    hit["YR"] = hit["SEASON"].str[:4].astype(int)
    d = hit["YR"].min()
    y12 = hit[hit["YR"] <= d + 1]
    mins, gp, age = y12["MIN"].sum(), y12["GP"].sum(), y12["AGE"].iloc[0]
    print(f"{name} is in the NBA data ({d}-{str(d + 1)[-2:]} debut, {mins:.0f} min "
          f"at {mins / gp:.1f} MPG over his first two seasons) but isn't scored:")
    if d < 2008:
        print("  debuted before the 2008 cohort window — established player.")
    elif age > 25:
        print(f"  age {age:.0f} at debut, over the <= 25 cohort limit.")
    elif mins / gp >= 12 and mins >= 500:
        print("  he already got a real early opportunity (12+ MPG on 500+ minutes).")
        print("  The model only scores end-of-bench players still waiting for one —")
        print("  a rotation-level rookie is not an under-the-radar case, by design.")
    else:
        print("  likely newer than the caches — rerun: python player_report.py --rebuild")


def report(query):
    if not UNIVERSE.exists():
        build_cache()
    allp = pd.read_csv(UNIVERSE)
    sv = pd.read_csv(SHAPVALS)

    keys = allp["PLAYER_NAME"].str.lower().tolist()
    contains = allp[allp["PLAYER_NAME"].str.lower().str.contains(query.lower())]
    if contains.empty:
        # in the NBA data but not scored? explain the exclusion honestly
        # (rotation player, established vet, too old at debut) — don't claim
        # "no NBA minutes" for a guy who plays 20 a night
        nba = pd.read_csv(lm.DATA / "season_totals.csv")
        hit = nba[nba["PLAYER_NAME"].str.lower().str.contains(query.lower())]
        if len(hit):
            explain_exclusion(hit)
            return
        # truly no NBA minutes — check pre-NBA sources before fuzzy-guessing,
        # so a G-League-only or draft-and-stash player gets his signal profile
        import pre_nba
        row, display, _ = pre_nba.build_row(query)
        if row is not None:
            print(f"{display} has no NBA minutes yet — out of the main universe.\n"
                  "Falling back to the pre-NBA signals model:\n")
            pre_nba.report(query)
            return
        match = get_close_matches(query.lower(), keys, n=1, cutoff=0.6)
        if not match:
            print(f'no match for "{query}" in NBA, G-League, or college data.\n'
                  "note: established players and pre-2008 debuts are out of "
                  "universe by design.")
            return
        print(f'(no exact match for "{query}" — closest scored player shown)\n')
        i = keys.index(match[0])
    else:
        i = contains.index[0]
    r, s = allp.loc[i], sv.loc[i]
    print(f"=== BREAKOUT REPORT: {r['PLAYER_NAME']} ===")
    print(f"{int(r['COHORT'])} debut | {r['ARCHETYPE']} | {r['PEDIGREE']} | "
          f"age {r['AGE']:.0f} at debut | pool: {r['POOL']}")
    if r["POOL"] == "historical":
        print(f"outcome KNOWN: {r['TIER']} — score below is retrospective")
    tier, detail = confidence(r)
    print(f"\nbreakout score: {r['PROB']:.2f} "
          f"(p{(allp['PROB'] < r['PROB']).mean() * 100:.0f} of all scored players)")
    print(f"confidence: {tier} — {detail}")

    # lower-is-better stats where a "low value favors" note would be noise
    INVERTED = {"PICK", "LANE_AGILITY_TIME", "THREE_QUARTER_SPRINT", "TOV36",
                "GLG_TOV36", "PROJ_TOV36", "AGE", "COLLEGE_AGE"}

    def line(f):
        val, pct = r[f], (allp[f] < r[f]).mean() * 100
        v = f"{val:.2f}" if abs(val) < 10 else f"{val:.0f}"
        # a low stat can push the score UP (e.g. low weight on a shooter) — say so
        note = ""
        if f not in INVERTED:
            if s[f] > 0 and pct < 40:
                note = "  — low value favors this profile"
            elif s[f] < 0 and pct > 60:
                note = "  — high value hurts this profile"
        return f"  {LABELS.get(f, f):34s} {v:>7s}  (p{pct:.0f} of universe){note}"

    order = s.abs().sort_values(ascending=False).index
    pos = [f for f in order if s[f] > 0 and pd.notna(r[f])][:4]
    neg = [f for f in order if s[f] < 0 and pd.notna(r[f])][:4]
    print("\nANALYTICAL STRENGTHS (pushing the score up)")
    print("\n".join(line(f) for f in pos) or "  none of note")
    print("\nANALYTICAL WEAKNESSES (pushing the score down)")
    print("\n".join(line(f) for f in neg) or "  none of note")

    print(f"\nHISTORICAL COMPS ({r['ARCHETYPE']}s, {r['COMP_BREAKOUT_RATE']:.0%} "
          f"of them broke out):\n  {r['COMPS']}")
    print("\ncaveat: ranking score, not a calibrated probability — trust the rank,")
    print("the percentile, and the comp rate. Small samples stay noisy by nature.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    elif sys.argv[1] == "--rebuild":
        build_cache()
    else:
        report(" ".join(sys.argv[1:]))
