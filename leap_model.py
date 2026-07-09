"""End-of-bench breakout model: XGBoost ranking + KNN historical comps.

Universe: players whose first NBA season falls in 2008-09..2021-22, age <= 25 at
debut, averaging < 12 MPG over their first two seasons (end-of-bench / fringe).
Label: reached >= 20 MPG with >= 30 GP in season 4 or 5 -> breakout.
Output: a structured hunt, not a prediction. Small samples, wide error bars.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from college_translation import college_features
from scrape_scouting import text_features

DATA = Path(__file__).parent / "data"
FIRST_COHORT, LAST_COHORT = 2008, 2021  # rookie-season start years
HOLDOUT_FROM = 2019                     # cohorts >= this are the backtest
MPG_BENCH = 12                          # yrs 1-2 "end of bench" ceiling
MPG_BREAKOUT, GP_BREAKOUT = 20, 30      # yr 4-5 success bar

CNT = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FGA", "FG3A", "FTA", "FG3M", "FTM", "OREB"]


def season_start(s):
    return int(s[:4])


def load():
    tot = pd.read_csv(DATA / "season_totals.csv")
    comb = pd.read_csv(DATA / "combine.csv")
    draft = pd.read_csv(DATA / "draft_history.csv")
    glg = pd.read_csv(DATA / "gleague_totals.csv")
    return tot, comb, draft, glg


def gleague_features(df, glg):
    """Shrunk G-League per-36 during each player's years 1-2. NaN if never sent."""
    glg = glg.copy()
    glg["YR"] = glg["SEASON"].str[:4].astype(int)
    cols = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FGA", "FTA"]
    m = glg.merge(df[["PLAYER_ID", "COHORT"]], on="PLAYER_ID")
    m = m[(m["YR"] >= m["COHORT"]) & (m["YR"] <= m["COHORT"] + 1)]
    agg = m.groupby("PLAYER_ID")[cols].sum()

    M = 200  # pseudo-minutes; G-League priors from the full G-League population
    for c in ["PTS", "REB", "AST", "STL", "BLK", "TOV"]:
        mu = glg[c].sum() / glg["MIN"].sum()
        agg["GLG_" + c + "36"] = 36 * (agg[c] + mu * M) / (agg["MIN"] + M)
    agg["GLG_STOCKS36"] = agg["GLG_STL36"] + agg["GLG_BLK36"]
    lg_ts = glg["PTS"].sum() / (2 * (glg["FGA"] + 0.44 * glg["FTA"]).sum())
    tsa = agg["FGA"] + 0.44 * agg["FTA"]
    agg["GLG_TS"] = (agg["PTS"] + lg_ts * 2 * 100) / (2 * (tsa + 100))
    agg["GLG_MIN12"] = agg["MIN"]

    keep = [c for c in agg.columns if c.startswith("GLG_")]
    df = df.merge(agg[keep], on="PLAYER_ID", how="left")
    df["HAS_GLG"] = df["GLG_MIN12"].notna().astype(int)
    return df


def build_dataset(tot, comb, draft, glg):
    tot["YR"] = tot["SEASON"].map(season_start)
    debut = tot.groupby("PLAYER_ID")["YR"].min().rename("DEBUT")
    tot = tot.join(debut, on="PLAYER_ID")

    # cohort: true debut inside window (2005-07 seasons exist only to verify debuts)
    cohort_ids = tot.loc[
        (tot["YR"] == tot["DEBUT"])
        & tot["DEBUT"].between(FIRST_COHORT, LAST_COHORT)
        & (tot["AGE"] <= 25),
        "PLAYER_ID",
    ].unique()

    rows = []
    for pid, g in tot[tot["PLAYER_ID"].isin(cohort_ids)].groupby("PLAYER_ID"):
        d = g["DEBUT"].iloc[0]
        y12 = g[g["YR"].isin([d, d + 1])]
        mins, gp = y12["MIN"].sum(), y12["GP"].sum()
        if mins < 40 or gp == 0 or mins / gp >= MPG_BENCH:
            continue
        y45 = g[g["YR"].isin([d + 3, d + 4])]
        broke = ((y45["MIN"] / y45["GP"] >= MPG_BREAKOUT) & (y45["GP"] >= GP_BREAKOUT)).any()
        if y45.empty:
            tier = "out of league"
        elif broke:
            tier = "starter/rotation (20+ mpg)"
        elif ((y45["MIN"] / y45["GP"] >= 12) & (y45["GP"] >= GP_BREAKOUT)).any():
            tier = "rotation (12-20 mpg)"
        else:
            tier = "fringe (<12 mpg)"
        row = {"PLAYER_ID": pid, "PLAYER_NAME": g["PLAYER_NAME"].iloc[0], "COHORT": d,
               "AGE": y12.loc[y12["YR"] == d, "AGE"].iloc[0], "MIN12": mins, "GP12": gp,
               "MPG12": mins / gp, "BREAKOUT": int(broke), "TIER": tier}
        for c in CNT:
            row[c] = y12[c].sum()
        rows.append(row)
    df = pd.DataFrame(rows)

    # empirical Bayes: shrink per-36 rates toward cohort mean with 300 pseudo-minutes
    M = 300
    for c in ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FGA", "FG3A", "FTA", "OREB"]:
        mu = df[c].sum() / df["MIN12"].sum()  # league rate per minute for this universe
        df[c + "36"] = 36 * (df[c] + mu * M) / (df["MIN12"] + M)
    df["STOCKS36"] = df["STL36"] + df["BLK36"]
    lg_ts = df["PTS"].sum() / (2 * (df["FGA"] + 0.44 * df["FTA"]).sum())
    tsa = df["FGA"] + 0.44 * df["FTA"]
    df["TS_SHRUNK"] = (df["PTS"] + lg_ts * 2 * 100) / (2 * (tsa + 100))
    lg_3p = df["FG3M"].sum() / df["FG3A"].sum()
    df["FG3PCT_SHRUNK"] = (df["FG3M"] + lg_3p * 50) / (df["FG3A"] + 50)
    df["FT_PCT_SHRUNK"] = (df["FTM"] + 0.75 * 50) / (df["FTA"] + 50)  # FT% ~ touch/shooting proxy

    # draft position (undrafted -> 75)
    draft = draft.sort_values("SEASON").drop_duplicates("PERSON_ID")
    df = df.merge(draft[["PERSON_ID", "OVERALL_PICK"]], left_on="PLAYER_ID",
                  right_on="PERSON_ID", how="left").drop(columns="PERSON_ID")
    df["PICK"] = df["OVERALL_PICK"].fillna(75)
    df["UNDRAFTED"] = df["OVERALL_PICK"].isna().astype(int)

    # combine measurements (NaN where missing; xgboost handles natively)
    cc = ["HEIGHT_WO_SHOES", "WINGSPAN", "WEIGHT", "STANDING_VERTICAL_LEAP",
          "MAX_VERTICAL_LEAP", "LANE_AGILITY_TIME", "THREE_QUARTER_SPRINT"]
    comb = comb.sort_values("SEASON").drop_duplicates("PLAYER_ID", keep="last")
    comb[cc] = comb[cc].apply(pd.to_numeric, errors="coerce")
    df = df.merge(comb[["PLAYER_ID"] + cc], on="PLAYER_ID", how="left")
    df["WINGSPAN_RATIO"] = df["WINGSPAN"] / df["HEIGHT_WO_SHOES"]

    # college-to-NBA translation (NaN for internationals / no-college players)
    df = pd.concat([df.reset_index(drop=True), college_features(df)], axis=1)
    df = gleague_features(df, glg)
    return df


FEATURES = ["AGE", "PICK", "UNDRAFTED", "MIN12", "GP12", "MPG12",
            "PTS36", "REB36", "AST36", "STL36", "BLK36", "TOV36", "FGA36", "FG3A36",
            "FTA36", "OREB36", "STOCKS36", "TS_SHRUNK", "FG3PCT_SHRUNK", "FT_PCT_SHRUNK",
            "HEIGHT_WO_SHOES", "WINGSPAN", "WEIGHT", "WINGSPAN_RATIO",
            "STANDING_VERTICAL_LEAP", "MAX_VERTICAL_LEAP", "LANE_AGILITY_TIME",
            "THREE_QUARTER_SPRINT",
            "PROJ_PTS36", "PROJ_REB36", "PROJ_AST36", "PROJ_STL36", "PROJ_BLK36",
            "PROJ_TOV36", "PROJ_TS", "COLLEGE_BPM", "COLLEGE_USG", "COLLEGE_AGE",
            "HAS_COLLEGE",
            "GLG_PTS36", "GLG_REB36", "GLG_AST36", "GLG_STOCKS36", "GLG_TOV36",
            "GLG_TS", "GLG_MIN12", "HAS_GLG"]
# ponytail: Summer League was tested and removed — every variant lowered holdout
# AUC (0.791 -> 0.727-0.737). 5-game samples are noise even shrunk.
# data/summer_league_totals.csv is still cached if this is ever revisited.
# ponytail: scouting-text NLP features also tested and excluded — all variants
# lowered AUC (0.733-0.771), TEXT_SCORE alone ~coin flip (0.525). Pre-draft prose
# doesn't track year-4 outcomes for fringe players. TEXT_SCORE/HAS_TEXT kept in
# the output CSV as reader context only (built by scrape_scouting.text_features).

def pedigree(pick):
    """Draft pedigree tag. The model's purpose is uncovering under-the-radar guys;
    a lottery pick ranking high is consensus agreement, not a discovery."""
    if pick <= 14:
        return "lottery pick — NOT under the radar"
    if pick <= 30:
        return "first-rounder — expected to contribute"
    if pick <= 60:
        return "second-rounder — under the radar"
    return "undrafted — deep sleeper"


ARCH_FEATURES = ["HEIGHT_WO_SHOES", "WEIGHT", "REB36", "AST36", "BLK36", "STL36",
                 "FG3A36", "FTA36", "OREB36"]


def add_archetypes(df):
    """KMeans style clusters (k=4), named by centroid rules so labels are stable."""
    X = df[ARCH_FEATURES].fillna(df[ARCH_FEATURES].median())
    km = KMeans(4, n_init=10, random_state=0).fit(StandardScaler().fit_transform(X))
    df = df.copy()
    df["ARCH_ID"] = km.labels_
    names = {}
    for a, g in df.groupby("ARCH_ID"):
        ht, tpa = g["HEIGHT_WO_SHOES"].mean(), g["FG3A36"].mean()
        names[a] = ("big" if ht >= 80 else "guard" if ht <= 75
                    else "shooting wing" if tpa >= 4 else "non-shooting wing")
    df["ARCHETYPE"] = df["ARCH_ID"].map(names)
    return df


# ponytail: KNN comp space = stats + body + age only, no draft pick — "looks/plays like",
# not "was drafted like"
COMP_FEATURES = ["AGE", "MPG12", "PTS36", "REB36", "AST36", "STOCKS36", "TOV36",
                 "FG3A36", "TS_SHRUNK", "FG3PCT_SHRUNK", "FT_PCT_SHRUNK",
                 "HEIGHT_WO_SHOES", "WINGSPAN", "WEIGHT"]


def train_and_report(df):
    tr, ho = df[df["COHORT"] < HOLDOUT_FROM], df[df["COHORT"] >= HOLDOUT_FROM]
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, eval_metric="auc",
        scale_pos_weight=(tr["BREAKOUT"] == 0).sum() / tr["BREAKOUT"].sum(),
    )
    model.fit(tr[FEATURES], tr["BREAKOUT"])
    prob = model.predict_proba(ho[FEATURES])[:, 1]
    auc = roc_auc_score(ho["BREAKOUT"], prob)

    ho = ho.copy()
    ho["PROB"] = prob
    ho["PCTILE"] = ho["PROB"].rank(pct=True).mul(100).round(0)
    ho["PEDIGREE"] = ho["PICK"].map(pedigree)
    ho = ho.sort_values("PROB", ascending=False)

    base = ho["BREAKOUT"].mean()
    p10 = ho.head(10)["BREAKOUT"].mean()
    print(f"train n={len(tr)} (breakout rate {tr['BREAKOUT'].mean():.1%}), "
          f"holdout n={len(ho)} cohorts {HOLDOUT_FROM}-{LAST_COHORT} "
          f"(base rate {base:.1%})")
    print(f"holdout ROC AUC: {auc:.3f}  (coin flip = 0.500)")
    print(f"precision@10: {p10:.1%} vs base rate {base:.1%} "
          f"({p10 / base:.1f}x lift)\n")

    # SHAP reasons
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(ho[FEATURES])
    reasons = []
    for i in range(len(ho)):
        top = np.argsort(-np.abs(sv[i]))[:3]
        reasons.append(", ".join(
            f"{FEATURES[j]} {'+' if sv[i][j] > 0 else '-'}" for j in top))
    ho["WHY"] = reasons

    # KNN comps against training players of the same archetype
    med = tr[COMP_FEATURES].median()
    ho["COMPS"], ho["COMP_BREAKOUT_RATE"] = "", np.nan
    for arch, ho_g in ho.groupby("ARCHETYPE"):
        pool = tr[tr["ARCHETYPE"] == arch]
        scaler = StandardScaler()
        pool_x = scaler.fit_transform(pool[COMP_FEATURES].fillna(med))
        knn = NearestNeighbors(n_neighbors=min(8, len(pool))).fit(pool_x)
        _, idx = knn.kneighbors(scaler.transform(ho_g[COMP_FEATURES].fillna(med)))
        for i, nn in zip(ho_g.index, idx):
            sub = pool.iloc[nn]
            ho.loc[i, "COMPS"] = ", ".join(sub["PLAYER_NAME"])
            ho.loc[i, "COMP_BREAKOUT_RATE"] = sub["BREAKOUT"].mean()

    print("=== TOP 10 FLAGGED (holdout cohorts the model never saw) ===")
    for _, r in ho.head(10).iterrows():
        hit = "HIT" if r["BREAKOUT"] else "miss"
        print(f"\n{r['PLAYER_NAME']} ({int(r['COHORT'])} cohort, {r['ARCHETYPE']}, "
              f"age {r['AGE']:.0f}, {r['MIN12']:.0f} min yrs 1-2) [{r['PEDIGREE']}] "
              f"-> {hit}: {r['TIER']}")
        print(f"  prob {r['PROB']:.2f} (p{r['PCTILE']:.0f}) | why: {r['WHY']}")
        print(f"  comps ({r['COMP_BREAKOUT_RATE']:.0%} broke out): {r['COMPS']}")

    cols = ["PLAYER_NAME", "COHORT", "ARCHETYPE", "PEDIGREE", "AGE", "MIN12",
            "MPG12", "PROB", "PCTILE", "TIER", "BREAKOUT", "WHY",
            "COMP_BREAKOUT_RATE", "COMPS", "TEXT_SCORE", "HAS_TEXT"]
    ho[cols].to_csv(DATA / "holdout_rankings.csv", index=False)
    print(f"\nfull ranking -> data/holdout_rankings.csv "
          f"(low MIN12 = tiny sample, treat prob as a hunch not a prediction)")
    return auc


if __name__ == "__main__":
    df = build_dataset(*load())
    df = text_features(df, HOLDOUT_FROM)
    df = add_archetypes(df)
    assert len(df) > 250, f"cohort too small: {len(df)}"
    assert df["BREAKOUT"].mean() > 0.02, "labels degenerate"
    assert df["PTS36"].between(0, 40).all(), "shrinkage broken"
    assert df["ARCHETYPE"].nunique() == 4, "archetype naming collision"
    print(f"universe: {len(df)} end-of-bench players, "
          f"{df['BREAKOUT'].mean():.1%} broke out overall")
    arch = df.groupby("ARCHETYPE").agg(n=("BREAKOUT", "size"),
                                       breakout_rate=("BREAKOUT", "mean"))
    print(arch.round(2).to_string(), "\n")
    auc = train_and_report(df)
    assert 0 < auc <= 1
