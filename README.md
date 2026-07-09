# NBA End-of-Bench Breakout Model

A model that tries to quantify a scout's gut feeling: given an end-of-bench NBA player —
the 11th man, the two-way guy, the fringe roster piece — what are the odds he becomes a
real rotation player if someone actually gives him minutes?

This is deliberately framed as a **structured hunch, not a prediction**. The samples are
tiny (that's the whole problem with evaluating bench players), the outcome is rare, and
real scouting judgment is noisy. The bar I set for the project was simple: **beat a coin
flip on a holdout of players the model never saw.** It clears that bar comfortably.

## The question, operationalized

- **Universe:** every player who debuted between 2008 and 2021, was 25 or younger at
  debut, and averaged **under 12 minutes per game across his first two seasons**
  (328 players).
- **Label:** did he reach **20+ MPG (30+ games) in his 4th or 5th season?**
  38 of 328 did (11.6%). 62% were out of the league entirely by then.
- **Backtest:** train on the 2008–2018 debut cohorts, test blind on 2019–2021.

**The "fallen angel" edge case.** A minutes-based universe has a blind spot: the player
who got early minutes he arguably hadn't earned — injuries, a tanking roster — and then
got buried without a real second look. He fails the under-12-MPG filter, but he was
never actually given a fair evaluation either. I measured the pattern rather than
guessing: 90 players since 2008 fit it (12–18 MPG in years 1–2, under 10 MPG by year 3),
and **6.7% broke out** — half the end-of-bench base rate, but the six hits include
Dejounte Murray. So these players get their own explicitly flagged pool on the current
board (`POOL = "fallen angel"`), scored by the same model but presented with their own
base rate. They are deliberately **not** added to the training universe — their
opportunity story is different, and the label would mean something different for them.

## Results

| Model version | Holdout ROC AUC |
|---|---|
| Coin flip | 0.500 |
| NBA stats only | 0.718 |
| + college-to-NBA translation | 0.746 |
| **+ G-League stats (final)** | **0.791** |

All 8 actual breakouts in the 105-player holdout rank in the model's top half; the #1
overall flag (Isaiah Joe) hit. Precision in the top 10 runs ~2.5–3x the base rate.

Two feature buckets were built, tested, and **rejected for making the model worse** —
the negative results are kept in the code as comments so they don't get re-tried:

- **Summer League stats** (AUC 0.791 → 0.73): five-game samples are noise even with
  heavy shrinkage.
- **Scouting-report NLP** (AUC 0.791 → 0.73–0.77): strengths/weaknesses text scraped
  for 300+ players, turned into trait-keyword features and a TF-IDF sentiment score.
  Pre-draft prose about fringe players is boilerplate and doesn't track year-4 outcomes.
  The text still rides along in the outputs as reading material.

## How it works

**Features** (~36 per player, three of the buckets survived testing):

1. **NBA production, shrunk.** Per-36 rates and shooting percentages from years 1–2,
   with empirical-Bayes shrinkage: every rate is padded with 300 pseudo-minutes of
   league-average production, so a 40-minute sample can't produce a wild point estimate.
2. **G-League production** during the same window (same treatment, 200 pseudo-minutes).
   The single most valuable addition — two-way players log real minutes there.
3. **College-to-NBA translation.** Ridge regressions (fit on 451 prior draftees, pre-2019
   only to keep the backtest clean) map final-college-season stats to expected NBA per-36.
   The regressions independently reproduce the classic Pelton-style finding: rebounding,
   blocks, and assists translate (R² 0.68–0.74); scoring efficiency barely does (R² ≤ 0.05).
4. **Bio and body:** age, draft pick, combine measurements, wingspan ratio.

**Models** (two, on purpose):

- **XGBoost classifier** with SHAP attribution — every score comes with its top-3
  reasons ("flagged because: young for level, G-League rim protection, efficient in
  limited minutes"), the way a scout would explain a hunch.
- **KNN historical comps** — the "reminds me of X" model. Standardized stat/body/age
  features, nearest 8 historical players *within the same archetype*, and their actual
  outcome distribution. When the two models disagree, believe the disagreement: a high
  GBM score with a 0% comp group is a red flag on the model itself.

**Archetypes:** KMeans (k=4) on style and size — big / guard / shooting wing /
non-shooting wing. Not fed to the classifier; used to constrain comps and as an
analytical lens. Sharpest single insight in the project: end-of-bench **bigs break out
at 16%, non-shooting wings at 6%**. The tweener with no jumper really is a dead end.

**Pedigree flag:** every output row is tagged lottery pick / first-rounder /
second-rounder / undrafted. The model's purpose is finding *under-the-radar* players —
a lottery pick scoring high is consensus agreement, not a discovery. The real
deliverable is the undrafted guy with a 90th-percentile profile.

What the model learned to weight, in scout language: *efficient in his minutes, shoots
or protects the rim, young for the level, productive in the bigger G-League sample.*
Nobody told it that; it's what survives contact with 14 years of outcomes.

## Running it

```bash
pip install nba_api xgboost shap scikit-learn pandas requests beautifulsoup4

python pull_data.py        # pulls & caches all raw data into data/ (~5 min, resumable)
python scrape_scouting.py  # optional: scouting text from NBADraft.net (~6 min)
python leap_model.py       # backtest: trains, evaluates on holdout, writes rankings
python score_current.py    # the live board: scores current-era bench players

python player_report.py "isaiah joe"   # look up any scored player by name
```

The player lookup prints a full report: breakout score and percentile, a sample-size
confidence tier (how many NBA + G-League minutes back the numbers), analytical
strengths and weaknesses (the SHAP factors pushing the score up or down, each shown
with the underlying stat and its percentile in the universe), and the 8 nearest
historical comps with their real outcome rate. Fuzzy name matching included; the first
run builds a cache, lookups after that are instant (`--rebuild` after refreshing data).

| File | What it does |
|---|---|
| `pull_data.py` | NBA season totals 2005–26, G-League (same API, `league_id="20"`), Barttorvik college CSVs 2008–26, combine, draft history. Everything cached to `data/`. |
| `college_translation.py` | The college→NBA ridge translation. Runnable standalone to see the fit. |
| `scrape_scouting.py` | Scouting text scrape + NLP features (kept for context columns despite the negative result). |
| `leap_model.py` | Universe, labels, features, archetypes, XGBoost + SHAP, comps, backtest → `data/holdout_rankings.csv` |
| `player_report.py` | Name-lookup interface: score, confidence tier, analytical strengths/weaknesses, comps for any scored player |
| `score_current.py` | Scores unlabeled 2022–25 debuts with the model trained on all 328 labeled players → `data/current_board.csv`. Includes a second, flagged pool: **"fallen angels"** — players who got early circumstantial minutes (12–18 MPG in years 1–2) and have been buried since. Historically they break out at 6.7% vs 11.6% for the true end-of-bench pool (but that 6.7% includes Dejounte Murray). |

Output rows include: score, percentile, archetype, pedigree tag, SHAP reasons, the 8
comps and their breakout rate, and sample-size context (minutes played).

## Honest limitations

- **The score is a ranking, not a calibrated probability.** Class reweighting inflates
  it; trust the rank, the percentile, and the comp group's breakout rate.
- **38 positive labels.** Every metric has wide error bars. Precision@10 moves 10 points
  per player.
- The label is minutes-based; a player can "break out" into bad minutes on a tanking
  team, and a good player can be buried behind a star. On/off or EPM-based labels would
  be better but need play-by-play.
- Garbage-time minutes aren't filtered (shrinkage partially compensates).
- In the fallen-angel pool, "0 MPG now" can't distinguish *buried* from *out of the
  league* or *retired* — read those rows with that in mind.
- Data comes from public endpoints (stats.nba.com via `nba_api`, barttorvik.com,
  nbadraft.net) pulled politely with caching. `data/` is not committed — run
  `pull_data.py` to rebuild it locally.

## Ideas that didn't make the cut / future work

- Garbage-time-filtered splits (needs play-by-play; biggest remaining data upgrade)
- On/off- or EPM-based outcome labels
- Team-context weighting of early minutes (minutes on a 25-win team ≠ minutes on a
  contender) — would separate true "circumstantial minutes" fallen angels from players
  who earned a look and lost it
- Qualitative analysis done right: draft reports, beat-writer coverage, and
  intangibles reporting (work ethic, coachability, role acceptance) as structured
  inputs on player tendency to break out. The first NLP attempt (pre-draft
  strengths/weaknesses boilerplate) didn't carry signal, but richer qualitative
  sources — reporting on how a player is developing *after* reaching the league —
  remain the most scout-like information the model doesn't yet see
- Draft big-board ranks as consensus-vs-model context on the current board
  (`score_current.py` already has the merge slot: `data/athletic_boards.csv`)
