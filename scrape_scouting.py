"""Scouting-text NLP bucket: scrape NBADraft.net strengths/weaknesses profiles,
turn them into features. ~50% coverage of the fringe universe is expected —
these are guys scouts barely wrote about; HAS_TEXT carries that signal too.
"""
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

DATA = Path(__file__).parent / "data"
UA = {"User-Agent": "Mozilla/5.0"}
END_MARKERS = ["Notes:", "Overall:", "NBA Comparison", "Related News", "Aran Smith",
               "Notes ", "YouTube"]

# scout-speak trait lexicon, grouped so we get a few dense features, not 30 sparse ones
TRAITS = {
    "TXT_MOTOR": ["motor", "work ethic", "hard worker", "works hard", "hustle",
                  "energy", "relentless", "competes", "competitor"],
    "TXT_FEEL": ["feel for the game", "basketball iq", "instincts", "coachable",
                 "high iq", "smart", "savvy", "fundamentals", "unselfish"],
    "TXT_TOOLS": ["athletic", "athleticism", "explosive", "wingspan", "length",
                  "quick", "leaper", "bounce", "speed"],
    "TXT_UPSIDE": ["upside", "potential", "ceiling", "intriguing", "raw",
                   "project", "developing"],
    "TXT_RED_FLAGS": ["lazy", "questionable", "inconsistent", "struggles", "lacks",
                      "soft", "undersized", "tweener", "concern", "limited",
                      "below average", "poor"],
}


def slugify(name):
    return re.sub(r"[^a-z ]", "", name.lower()).strip().replace(" ", "-")


def fetch_profile(name):
    url = f"https://www.nbadraft.net/players/{slugify(name)}/"
    try:
        r = requests.get(url, headers=UA, timeout=20)
    except requests.RequestException:
        return "", ""
    if r.status_code != 200:
        return "", ""
    txt = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
    m = re.search(r"Strengths:?(.*?)Weaknesses:?", txt, re.S)
    if not m:
        return "", ""
    strengths = m.group(1)
    rest = txt[m.end():]
    end = min([rest.find(e) for e in END_MARKERS if rest.find(e) > 0] + [3000])
    return strengths[:3000].strip(), rest[:end].strip()


def scrape_all(names):
    out = DATA / "scouting_text.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["PLAYER_NAME", "STRENGTHS", "WEAKNESSES"])
    todo = [n for n in names if n not in set(done["PLAYER_NAME"])]
    rows = []
    for i, n in enumerate(todo):
        s, w = fetch_profile(n)
        rows.append({"PLAYER_NAME": n, "STRENGTHS": s, "WEAKNESSES": w})
        if s:
            print(f"[{i + 1}/{len(todo)}] {n}: {len(s)}+{len(w)} chars")
        time.sleep(0.8)
    pd.concat([done, pd.DataFrame(rows)]).to_csv(out, index=False)


def text_features(df, holdout_from):
    """Merge NLP features onto cohort df. TEXT_SCORE is a TF-IDF logistic score:
    out-of-fold for train cohorts, fit-on-train for holdout (no leakage)."""
    tx = pd.read_csv(DATA / "scouting_text.csv").fillna("")
    tx["DOC"] = (tx["STRENGTHS"] + " " + tx["WEAKNESSES"]).str.lower()
    df = df.merge(tx, on="PLAYER_NAME", how="left")
    df["DOC"] = df["DOC"].fillna("")
    df["HAS_TEXT"] = (df["DOC"].str.len() > 100).astype(int)

    words = df["DOC"].str.split().str.len().clip(lower=1)
    for feat, terms in TRAITS.items():
        hits = sum(df["DOC"].str.count(re.escape(t)) for t in terms)
        df[feat] = np.where(df["HAS_TEXT"] == 1, 1000 * hits / words, np.nan)
    sl, wl = df["STRENGTHS"].fillna("").str.len(), df["WEAKNESSES"].fillna("").str.len()
    df["TXT_STR_RATIO"] = np.where(df["HAS_TEXT"] == 1, sl / (sl + wl + 1), np.nan)

    # TF-IDF -> logistic regression "scout sentiment" score
    df["TEXT_SCORE"] = np.nan
    has = df["HAS_TEXT"] == 1
    tr = has & (df["COHORT"] < holdout_from)
    ho = has & (df["COHORT"] >= holdout_from)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=2000,
                          stop_words="english", sublinear_tf=True)
    Xtr = vec.fit_transform(df.loc[tr, "DOC"])
    lr = LogisticRegression(C=0.1, max_iter=1000)
    df.loc[tr, "TEXT_SCORE"] = cross_val_predict(
        lr, Xtr, df.loc[tr, "BREAKOUT"], cv=5, method="predict_proba")[:, 1]
    lr.fit(Xtr, df.loc[tr, "BREAKOUT"])
    df.loc[ho, "TEXT_SCORE"] = lr.predict_proba(vec.transform(df.loc[ho, "DOC"]))[:, 1]
    return df.drop(columns=["STRENGTHS", "WEAKNESSES", "DOC"])


TEXT_FEATURES = list(TRAITS) + ["TXT_STR_RATIO", "TEXT_SCORE", "HAS_TEXT"]

if __name__ == "__main__":
    import leap_model  # runtime import; avoids circularity at module load
    df = leap_model.build_dataset(*leap_model.load())
    scrape_all(df["PLAYER_NAME"].tolist())
    tx = pd.read_csv(DATA / "scouting_text.csv").fillna("")
    n = (tx["STRENGTHS"].str.len() > 100).sum()
    print(f"\ncoverage: {n}/{len(df)} players with real scouting text")
    assert n > 80, "coverage too low to be useful"
