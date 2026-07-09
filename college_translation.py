"""College->NBA statistical translation (Pelton-style, minimal version).

Ridge regressions from final-college-season stats to NBA years-1-2 per-36
production. Fit on debutants < 2019 with >= 500 NBA minutes (stable targets,
no holdout leakage); predictions generated for every matched player.
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "data"
POWER_CONFS = {"ACC", "B10", "B12", "BE", "SEC", "P12", "P10"}

COLLEGE_X = ["usg", "ts_pct", "orb_pct", "drb_pct", "ast_pct", "to_pct", "blk_pct",
             "stl_pct", "tp_pct_shrunk", "ftr", "bpm", "height_in", "college_age",
             "power_conf"]
TARGETS = ["PTS36", "REB36", "AST36", "STL36", "BLK36", "TOV36", "TS"]


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower())
    return re.sub(r" (jr|sr|ii|iii|iv)$", "", s.strip())


def height_inches(h):
    try:
        f, i = str(h).split("-")
        return int(f) * 12 + int(i)
    except (ValueError, AttributeError):
        return np.nan


def college_table():
    """One row per (name key, season): final-season predictors."""
    t = pd.read_csv(DATA / "barttorvik.csv", low_memory=False)
    t["key"] = t["player_name"].map(norm_name)
    t["height_in"] = t["height"].map(height_inches)
    t["power_conf"] = t["conf"].isin(POWER_CONFS).astype(int)
    bd = pd.to_datetime(t["birthdate"], errors="coerce")
    season_end = pd.to_datetime(t["year"].astype(str) + "-04-01")
    t["college_age"] = (season_end - bd).dt.days / 365.25
    lg3 = t["tpm"].sum() / t["tpa"].sum()
    t["tp_pct_shrunk"] = (t["tpm"] + lg3 * 50) / (t["tpa"] + 50)
    return t[["key", "year", "GP", "mpg"] + COLLEGE_X]


def match_college(names_cohorts):
    """names_cohorts: df with PLAYER_NAME, COHORT. Returns matched college rows:
    the player's last college season ending no later than debut year + 1."""
    t = college_table()
    t = t[(t["GP"] >= 10) & (t["mpg"] >= 10)]  # skip cameo seasons
    df = names_cohorts.copy()
    df["key"] = df["PLAYER_NAME"].map(norm_name)
    m = df.merge(t, on="key", how="left")
    m = m[m["year"].isna() | (m["year"] <= m["COHORT"] + 1)]
    m = m.sort_values("year").groupby("key", as_index=False).last()
    return df.merge(m[["key"] + COLLEGE_X], on="key", how="left")


def nba_y12_targets():
    """Years-1-2 NBA per-36 production for all 2008-2018 debutants, >=500 min."""
    tot = pd.read_csv(DATA / "season_totals.csv")
    tot["YR"] = tot["SEASON"].str[:4].astype(int)
    tot["DEBUT"] = tot.groupby("PLAYER_ID")["YR"].transform("min")
    tot = tot[tot["DEBUT"].between(2008, 2018) & (tot["YR"] <= tot["DEBUT"] + 1)]
    g = tot.groupby("PLAYER_ID").agg(
        PLAYER_NAME=("PLAYER_NAME", "first"), COHORT=("DEBUT", "first"),
        MIN=("MIN", "sum"), PTS=("PTS", "sum"), REB=("REB", "sum"),
        AST=("AST", "sum"), STL=("STL", "sum"), BLK=("BLK", "sum"),
        TOV=("TOV", "sum"), FGA=("FGA", "sum"), FTA=("FTA", "sum"))
    g = g[g["MIN"] >= 500].reset_index()
    for c in ["PTS", "REB", "AST", "STL", "BLK", "TOV"]:
        g[c + "36"] = 36 * g[c] / g["MIN"]
    g["TS"] = g["PTS"] / (2 * (g["FGA"] + 0.44 * g["FTA"]))
    return g


def fit_translation(verbose=True):
    """Returns dict of fitted pipelines, one per target stat."""
    fit = match_college(nba_y12_targets())
    fit = fit.dropna(subset=["bpm"])  # college match required to fit
    models = {}
    for tgt in TARGETS:
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             Ridge(alpha=10))
        cv = cross_val_score(pipe, fit[COLLEGE_X], fit[tgt], cv=5, scoring="r2")
        pipe.fit(fit[COLLEGE_X], fit[tgt])
        models[tgt] = pipe
        if verbose:
            print(f"  {tgt}: CV R2 = {cv.mean():.2f} (n={len(fit)})")
    return models


def college_features(names_cohorts):
    """PROJ_* translated per-36 projections + raw college signal, merged onto
    a df with PLAYER_NAME and COHORT. NaN where no college match."""
    models = fit_translation()
    out = match_college(names_cohorts)
    matched = out["bpm"].notna()
    for tgt, pipe in models.items():
        out["PROJ_" + tgt] = np.nan
        out.loc[matched, "PROJ_" + tgt] = pipe.predict(out.loc[matched, COLLEGE_X])
    out = out.rename(columns={"bpm": "COLLEGE_BPM", "usg": "COLLEGE_USG",
                              "college_age": "COLLEGE_AGE"})
    out["HAS_COLLEGE"] = matched.astype(int)
    keep = (["PROJ_" + t for t in TARGETS]
            + ["COLLEGE_BPM", "COLLEGE_USG", "COLLEGE_AGE", "HAS_COLLEGE"])
    return out[keep]


if __name__ == "__main__":
    models = fit_translation()
    demo = pd.DataFrame({"PLAYER_NAME": ["Stephen Curry"], "COHORT": [2009]})
    f = college_features(demo)
    assert f["HAS_COLLEGE"].iloc[0] == 1
    assert 10 < f["PROJ_PTS36"].iloc[0] < 30, f["PROJ_PTS36"].iloc[0]
    print("\nCurry 2009 translated:", f.round(2).to_dict("records")[0])
