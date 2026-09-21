# Result (2026-09-21): 193 cell x bucket combinations; 26 with |t|>=2 vs 18.2+/-2.8 under
# shuffled labels; 0 "found on train, held on test" vs 0.1 expected. No condition (volume
# ratio, RSI, ATR, price band, vs EMA20, ^AXJO 20d regime, ASIC short level/change) turns
# any setup into a signal. macd_hist_up/confirmed is consistently NEGATIVE vs the universe
# (test t -3 to -4) but nets +0.008% as a short after costs. Low-ATR longs look bad only
# because the equal-weight control is small-cap heavy (size effect, not setup effect).
# Announcement/classifier conditions could not be tested: that data starts 2026-08-19.

"""Conditioning study: does any context turn a /setups detection into a signal?

Population: backfill detections (first_in_window=1), outcome = direction-signed
5d return minus the same-day equal-weight universe. For every setup x state cell
and every condition bucket: n, mean excess, date-clustered t, hit rate. Found on
train (first 2/3 of dates), confirmed on test (last 1/3). A shuffled-label
control says how many hits to expect by chance.
"""
import json, math, sqlite3, sys
from bisect import bisect_right
import numpy as np, pandas as pd

MIN_N_BUCKET = 150
MIN_DATES = 40

sw = sqlite3.connect("/opt/tradingagents/data/swing.db")
df = pd.read_sql("""
SELECT d.scan_date, d.ticker, d.setup_id, d.state, d.direction, d.tier, d.close_eod AS price,
       d.vol_ratio_eod, d.context_json, d.fwd_5d_pct, d.fwd_10d_pct, d.mfe_10d_pct, d.mae_10d_pct,
       c.ew_fwd_5d_pct, c.ew_fwd_10d_pct
FROM ta_detections d LEFT JOIN ta_control c ON c.date = d.scan_date
WHERE d.source='backfill' AND d.first_in_window=1 AND d.fwd_5d_pct IS NOT NULL AND c.ew_fwd_5d_pct IS NOT NULL
""", sw)
ctx = pd.DataFrame([json.loads(x or "{}") for x in df.context_json])
for k in ("rsi", "atr_pct", "ema20", "bb_pct_b"):
    df[k] = ctx[k].astype(float)
df["ex5"] = df.direction * (df.fwd_5d_pct - df.ew_fwd_5d_pct)
df["ex10"] = df.direction * (df.fwd_10d_pct - df.ew_fwd_10d_pct)
df["cell"] = df.setup_id + "/" + df.state
print(f"population {len(df)} rows, {df.cell.nunique()} cells, {df.scan_date.nunique()} dates", file=sys.stderr)

# --- market regime: ^AXJO 20d trailing return -------------------------------
import yfinance as yf
ax = yf.download("^AXJO", period="4y", interval="1d", progress=False, auto_adjust=True)["Close"]
ax = ax.squeeze()
ax.index = [d.date().isoformat() for d in ax.index]
reg = (ax / ax.shift(20) - 1) * 100
df["mkt20"] = df.scan_date.map(reg)

# --- ASIC short interest, point-in-time on publication_date ------------------
asx = sqlite3.connect("file:/opt/asxbrief/data/asx.db?mode=ro", uri=True)
tick = tuple(sorted(df.ticker.unique()))
sp = pd.read_sql(f"""SELECT ticker, position_date, publication_date, pct_short FROM short_positions
                     WHERE ticker IN ({','.join('?'*len(tick))}) AND position_date >= '2023-05-01'""",
                 asx, params=tick)
sp = sp.sort_values(["ticker", "publication_date"])
short_level, short_chg = {}, {}
for t, g in sp.groupby("ticker"):
    pub = g.publication_date.tolist(); pos = g.position_date.tolist(); v = g.pct_short.to_numpy(float)
    short_level[t] = (pub, v, pos)
def lookup(t, date):
    r = short_level.get(t)
    if not r: return (np.nan, np.nan)
    pub, v, pos = r
    i = bisect_right(pub, date) - 1
    if i < 0: return (np.nan, np.nan)
    j = i - 20
    return (v[i], v[i] - v[j] if j >= 0 else np.nan)
vals = [lookup(t, d) for t, d in zip(df.ticker, df.scan_date)]
df["short_pct"] = [a for a, b in vals]; df["short_chg20"] = [b for a, b in vals]

# --- condition buckets -------------------------------------------------------
def q3(s):
    return pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"])
conds = {
    "vol_ratio": pd.cut(df.vol_ratio_eod, [0, 1, 2, 3, 1e9], labels=["<1x", "1-2x", "2-3x", ">=3x"]),
    "rsi": pd.cut(df.rsi, [0, 30, 45, 55, 70, 100], labels=["<30", "30-45", "45-55", "55-70", ">70"]),
    "atr_pct": q3(df.atr_pct),
    "price": pd.cut(df.price, [0, 2, 10, 1e9], labels=["<$2", "$2-10", ">$10"]),
    "vs_ema20": np.where(df.price > df.ema20, "above", "below"),
    "mkt20": pd.cut(df.mkt20, [-99, -3, 3, 99], labels=["mkt<-3%", "mkt flat", "mkt>+3%"]),
    "short_pct": pd.cut(df.short_pct, [-1, 1, 3, 6, 100], labels=["<1%", "1-3%", "3-6%", ">6%"]),
    "short_chg20": pd.cut(df.short_chg20, [-99, -0.5, 0.5, 99], labels=["falling", "flat", "rising"]),
}
for k, v in conds.items():
    df[f"c_{k}"] = pd.Series(v, index=df.index).astype(object)

split = np.sort(df.scan_date.unique())[int(df.scan_date.nunique() * 2 / 3)]
df["train"] = df.scan_date < split

def stats(g, col="ex5"):
    if len(g) < MIN_N_BUCKET: return None
    dm = g.groupby("scan_date")[col].mean()
    if len(dm) < MIN_DATES: return None
    sd = dm.std(ddof=1)
    t = dm.mean() / (sd / math.sqrt(len(dm))) if sd > 0 else 0.0
    return {"n": len(g), "dates": len(dm), "mean": g[col].mean(), "t": t, "hit": (g[col] > 0).mean()}

def run(frame, label_col_suffix=""):
    rows = []
    for cell, g in frame.groupby("cell"):
        base = stats(g)
        for k in conds:
            col = f"c_{k}{label_col_suffix}"
            for b, gb in g.groupby(col):
                tr, te = stats(gb[gb.train]), stats(gb[~gb.train])
                if not tr or not te: continue
                al = stats(gb); al10 = stats(gb, "ex10")
                rows.append({"cell": cell, "cond": k, "bucket": b, "n": al["n"], "dates": al["dates"],
                             "mean5": al["mean"], "t5": al["t"], "hit": al["hit"],
                             "mean10": al10["mean"] if al10 else np.nan, "t10": al10["t"] if al10 else np.nan,
                             "train_t": tr["t"], "test_t": te["t"], "test_mean": te["mean"],
                             "base_mean": base["mean"] if base else np.nan})
    return pd.DataFrame(rows)

res = run(df)
res["confirmed"] = (res.train_t >= 2.0) & (res.test_t >= 1.0) & (res.test_mean > 0)
print(f"\n{len(res)} (cell x bucket) combinations tested; {int((res.t5.abs() >= 2).sum())} with |t|>=2 overall; "
      f"{int(res.confirmed.sum())} 'found on train (t>=2) and held on test (t>=1)'")

# --- shuffled-label control --------------------------------------------------
rng = np.random.default_rng(7)
ctrl_hits, ctrl_conf = [], []
for rep in range(20):
    sh = df.copy()
    for k in conds:
        sh[f"c_{k}_sh"] = sh.groupby("cell")[f"c_{k}"].transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).to_numpy())
    r = run(sh, "_sh")
    ctrl_hits.append(int((r.t5.abs() >= 2).sum()))
    ctrl_conf.append(int(((r.train_t >= 2.0) & (r.test_t >= 1.0) & (r.test_mean > 0)).sum()))
print(f"shuffled-label control (20 reps): |t|>=2 combos {np.mean(ctrl_hits):.1f} +/- {np.std(ctrl_hits):.1f}; "
      f"'found and held' {np.mean(ctrl_conf):.1f} +/- {np.std(ctrl_conf):.1f}")

pd.set_option("display.width", 200)
cols = ["cell", "cond", "bucket", "n", "dates", "base_mean", "mean5", "t5", "hit", "mean10", "t10", "train_t", "test_t", "test_mean"]
print("\n=== top 20 by overall t (5d excess) ===")
print(res.sort_values("t5", ascending=False)[cols].head(20).round(2).to_string(index=False))
print("\n=== 'found on train, held on test' ===")
print(res[res.confirmed].sort_values("test_t", ascending=False)[cols].round(2).to_string(index=False))
print("\n=== bottom 10 (conditions that HURT) ===")
print(res.sort_values("t5")[cols].head(10).round(2).to_string(index=False))
res.to_csv("/tmp/claude-0/-root/bd8892e7-ae5b-481c-bd20-9fb12f233504/scratchpad/condition_results.csv", index=False)
