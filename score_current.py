"""Score current (unlabeled) end-of-bench players: debut cohorts 2022-2025.

Trains on all 328 labeled players (2008-2021 cohorts), outputs
data/current_board.csv. If data/athletic_boards.csv exists (columns:
PLAYER_NAME, BOARD_RANK, BOARD_YEAR — one row per player per board), attaches
consensus-scout context: board rank and board-vs-pick slide.
2022 debuts are ~80% through their outcome window; 2024-25 debuts are pure
projection. This is a hunch board, not a prediction sheet.
"""
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import leap_model as lm
from college_translation import norm_name
from scrape_scouting import text_features

CUR_FIRST, CUR_LAST = 2022, 2025


def build():
    data = lm.load()
    labeled = lm.build_dataset(*data)
    # ponytail: monkeypatch cohort window + widen the MPG ceiling to also catch
    # "fallen angels" — early circumstantial minutes (12-18 mpg), buried since.
    # Historical breakout rates: end-of-bench 11.6%, fallen angels 6.7%.
    lm.FIRST_COHORT, lm.LAST_COHORT, lm.MPG_BENCH = CUR_FIRST, CUR_LAST, 18
    current = lm.build_dataset(*data)

    tot = data[0]
    last = tot[tot["YR"] == tot["YR"].max()].groupby("PLAYER_ID").agg(
        LMIN=("MIN", "sum"), LGP=("GP", "sum"))
    current = current.join((last["LMIN"] / last["LGP"]).rename("LATEST_MPG"),
                           on="PLAYER_ID")
    current["LATEST_MPG"] = current["LATEST_MPG"].fillna(0)
    current["POOL"] = np.where(current["MPG12"] < 12, "end-of-bench",
                      np.where(current["LATEST_MPG"] < 10, "fallen angel", "drop"))
    current = current[current["POOL"] != "drop"]  # 12-18 mpg AND still playing = just a rotation guy

    both = pd.concat([labeled.assign(SET="train"), current.assign(SET="score")],
                     ignore_index=True)
    both = text_features(both, holdout_from=CUR_FIRST)  # text model fit on labeled only
    both = lm.add_archetypes(both)  # one clustering for both, consistent names
    return both[both["SET"] == "train"].copy(), both[both["SET"] == "score"].copy()


def score():
    tr, cur = build()
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, eval_metric="auc",
        scale_pos_weight=(tr["BREAKOUT"] == 0).sum() / tr["BREAKOUT"].sum(),
    )
    model.fit(tr[lm.FEATURES], tr["BREAKOUT"])
    cur["PROB"] = model.predict_proba(cur[lm.FEATURES])[:, 1]
    cur["PCTILE"] = cur["PROB"].rank(pct=True).mul(100).round(0)
    cur["PEDIGREE"] = cur["PICK"].map(lm.pedigree)

    sv = shap.TreeExplainer(model).shap_values(cur[lm.FEATURES])
    cur["WHY"] = [", ".join(f"{lm.FEATURES[j]} {'+' if sv[i][j] > 0 else '-'}"
                            for j in np.argsort(-np.abs(sv[i]))[:3])
                  for i in range(len(cur))]

    # within-archetype comps against all labeled history
    med = tr[lm.COMP_FEATURES].median()
    cur["COMPS"], cur["COMP_BREAKOUT_RATE"] = "", np.nan
    for arch, g in cur.groupby("ARCHETYPE"):
        pool = tr[tr["ARCHETYPE"] == arch]
        scaler = StandardScaler()
        knn = NearestNeighbors(n_neighbors=8).fit(
            scaler.fit_transform(pool[lm.COMP_FEATURES].fillna(med)))
        _, idx = knn.kneighbors(scaler.transform(g[lm.COMP_FEATURES].fillna(med)))
        for i, nn in zip(g.index, idx):
            cur.loc[i, "COMPS"] = ", ".join(pool.iloc[nn]["PLAYER_NAME"])
            cur.loc[i, "COMP_BREAKOUT_RATE"] = pool.iloc[nn]["BREAKOUT"].mean()

    # Athletic big-board context, if provided
    boards = lm.DATA / "athletic_boards.csv"
    if boards.exists():
        b = pd.read_csv(boards)
        b["key"] = b["PLAYER_NAME"].map(norm_name)
        b = b.sort_values("BOARD_RANK").drop_duplicates("key")
        cur["key"] = cur["PLAYER_NAME"].map(norm_name)
        cur = cur.merge(b[["key", "BOARD_RANK", "BOARD_YEAR"]], on="key", how="left")
        cur["BOARD_VS_PICK"] = cur["PICK"] - cur["BOARD_RANK"]  # >0 = draft-night slide
    else:
        cur["BOARD_RANK"] = cur["BOARD_VS_PICK"] = np.nan
        print("(no data/athletic_boards.csv yet — board columns empty)")

    cur = cur.sort_values("PROB", ascending=False)
    cols = ["PLAYER_NAME", "COHORT", "ARCHETYPE", "PEDIGREE", "POOL", "AGE", "MIN12",
            "MPG12", "PROB", "PCTILE", "WHY", "COMP_BREAKOUT_RATE", "COMPS",
            "BOARD_RANK", "BOARD_VS_PICK", "GLG_MIN12", "TEXT_SCORE", "HAS_TEXT"]
    cur[cols].to_csv(lm.DATA / "current_board.csv", index=False)

    bench, fallen = cur[cur["POOL"] == "end-of-bench"], cur[cur["POOL"] == "fallen angel"]
    print(f"scored {len(bench)} end-of-bench + {len(fallen)} fallen angels "
          f"(cohorts {CUR_FIRST}-{CUR_LAST})\n")
    print("=== TOP 15 — CURRENT LEAP CANDIDATES (end-of-bench, base rate 11.6%) ===")
    for _, r in bench.head(15).iterrows():
        board = f" | board #{r['BOARD_RANK']:.0f}" if pd.notna(r["BOARD_RANK"]) else ""
        print(f"\n{r['PLAYER_NAME']} ({int(r['COHORT'])} debut, {r['ARCHETYPE']}, "
              f"age {r['AGE']:.0f}, {r['MIN12']:.0f} min) [{r['PEDIGREE']}]{board}")
        print(f"  prob {r['PROB']:.2f} (p{r['PCTILE']:.0f}) | {r['WHY']}")
        print(f"  comps ({r['COMP_BREAKOUT_RATE']:.0%} broke out): {r['COMPS']}")

    print("\n=== TOP 5 — FALLEN ANGELS (early circumstantial minutes, buried since; "
          "historical base rate 6.7%) ===")
    for _, r in fallen.head(5).iterrows():
        print(f"{r['PLAYER_NAME']} ({int(r['COHORT'])} debut, {r['ARCHETYPE']}, "
              f"age {r['AGE']:.0f}, {r['MPG12']:.1f} mpg yrs 1-2 -> "
              f"{r['LATEST_MPG']:.1f} now) [{r['PEDIGREE']}] prob {r['PROB']:.2f}")
    return cur


if __name__ == "__main__":
    cur = score()
    assert len(cur) > 50 and cur["PROB"].between(0, 1).all()
    print(f"\nfull board -> data/current_board.csv")
