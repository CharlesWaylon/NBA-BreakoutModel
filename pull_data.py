"""Pull and cache NBA data: season player totals, draft combine, draft history."""
import io
import time
import urllib.request
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, draftcombinestats, drafthistory

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2005, 2026)]  # 2005-06 .. 2025-26


def pull_season_totals():
    out = DATA / "season_totals.csv"
    if out.exists():
        return
    frames = []
    for s in SEASONS:
        for attempt in range(3):
            try:
                df = leaguedashplayerstats.LeagueDashPlayerStats(
                    season=s, per_mode_detailed="Totals", timeout=60
                ).get_data_frames()[0]
                break
            except Exception as e:
                print(f"{s} attempt {attempt}: {e}")
                time.sleep(5)
        else:
            raise RuntimeError(f"failed {s}")
        df["SEASON"] = s
        frames.append(df)
        print(f"{s}: {len(df)} players")
        time.sleep(1)
    pd.concat(frames).to_csv(out, index=False)


def pull_gleague_totals():
    out = DATA / "gleague_totals.csv"
    if out.exists():
        return
    frames = []
    for s in SEASONS:
        try:
            df = leaguedashplayerstats.LeagueDashPlayerStats(
                season=s, per_mode_detailed="Totals", league_id_nullable="20", timeout=60
            ).get_data_frames()[0]
        except Exception as e:  # early seasons may not exist on the API
            print(f"gleague {s}: skipped ({e})")
            continue
        df["SEASON"] = s
        frames.append(df)
        print(f"gleague {s}: {len(df)} players")
        time.sleep(1)
    pd.concat(frames).to_csv(out, index=False)


# Barttorvik CSVs ship headerless; column order verified against known players.
# Trailing ambiguous columns are named col_NN — none are needed for the translation.
TORVIK_COLS = [
    "player_name", "team", "conf", "GP", "min_pct", "ortg", "usg", "efg", "ts_pct",
    "orb_pct", "drb_pct", "ast_pct", "to_pct", "ftm", "fta", "ft_pct",
    "twom", "twoa", "two_pct", "tpm", "tpa", "tp_pct", "blk_pct", "stl_pct", "ftr",
    "class_yr", "height", "jersey", "porpag", "adjoe", "pfr", "year", "torvik_pid",
    "hometown", "recruit_rank", "ast_to", "rim_m", "rim_a", "mid_m", "mid_a",
    "rim_pct", "mid_pct", "dunk_m", "dunk_a", "dunk_pct", "pick", "drtg", "adrtg",
    "dporpag", "stops", "bpm", "obpm", "dbpm", "gbpm", "mpg", "ogbpm", "dgbpm",
    "oreb_pg", "dreb_pg", "treb_pg", "ast_pg", "stl_pg", "blk_pg", "pts_pg",
    "position", "col_65", "birthdate",
]


def pull_barttorvik():
    out = DATA / "barttorvik.csv"
    if out.exists():
        return
    frames = []
    for year in range(2008, 2027):
        url = f"https://barttorvik.com/getadvstats.php?year={year}&csv=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", errors="replace")
        if not raw.strip():
            print(f"torvik {year}: empty, skipped")
            continue
        df = pd.read_csv(io.StringIO(raw), header=None, names=TORVIK_COLS)
        frames.append(df)
        print(f"torvik {year}: {len(df)} players")
        time.sleep(1)
    pd.concat(frames).to_csv(out, index=False)


def pull_combine():
    out = DATA / "combine.csv"
    if out.exists():
        return
    df = draftcombinestats.DraftCombineStats(season_all_time="All Time", timeout=60).get_data_frames()[0]
    df.to_csv(out, index=False)
    print(f"combine: {len(df)} rows")


def pull_draft_history():
    out = DATA / "draft_history.csv"
    if out.exists():
        return
    df = drafthistory.DraftHistory(timeout=60).get_data_frames()[0]
    df.to_csv(out, index=False)
    print(f"draft history: {len(df)} rows")


if __name__ == "__main__":
    pull_season_totals()
    pull_gleague_totals()
    pull_barttorvik()
    pull_combine()
    pull_draft_history()
    print("done")
