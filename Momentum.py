import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
PATH = "/Users/abhishekbarak/Downloads/Momentum strategyy-2.xlsx"
SHEET = "Momentum strategy"

DO_1M = True
DO_12M = True
DO_12M_1 = True

ROLLING_WINDOW = 12   # rolling Sharpe window for chart 4
NW_LAGS = 3           # Newey–West lags for monthly data
MAIN_STRAT = "LS_1M"  # which strategy to feature in chart 5

# =========================
# LOAD
# =========================
df = pd.read_excel(PATH, sheet_name=SHEET)

# Try to detect a date/month column (optional, only for plotting)
date_col = None
for c in df.columns:
    if isinstance(c, str) and ("date" in c.lower() or "month" in c.lower()):
        date_col = c
        break

dates_raw = None
if date_col is not None:
    dates_raw = pd.to_datetime(df[date_col], errors="coerce")

# =========================
# RETURNS (KEEP NUMERIC INDEX FOR ALL CALCS)
# =========================
rets = df[["Monthly Return (IND)", "Monthly Return (US)", "Monthly_Return (Japan)"]].copy()
rets.columns = ["IND", "US", "JAPAN"]
rets = rets.apply(pd.to_numeric, errors="coerce")
rets = rets.dropna(how="all").reset_index(drop=True)   # RangeIndex 0..n-1

n = len(rets)

# Build plot_index separately (do NOT use it for calculations)
if dates_raw is not None:
    dates_clean = dates_raw.reset_index(drop=True).iloc[:n]
    # If too many missing or duplicates, fall back to synthetic monthly index
    if dates_clean.notna().sum() < max(10, n * 0.5) or dates_clean.duplicated().any():
        plot_index = pd.period_range(start="2006-01", periods=n, freq="M").to_timestamp("M")
    else:
        plot_index = dates_clean
else:
    plot_index = pd.period_range(start="2006-01", periods=n, freq="M").to_timestamp("M")

# =========================
# HELPERS
# =========================
def cumret(x):
    return np.prod(1 + x) - 1

def max_drawdown(r):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return np.nan
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    return dd.min()

def drawdown_series(r):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return pd.Series(dtype=float)
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1

def stats(r):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return pd.Series({
            "n": 0, "mean_m": np.nan, "median_m": np.nan, "std_m": np.nan,
            "ann_mean": np.nan, "ann_vol": np.nan, "sharpe_0rf": np.nan,
            "win_rate": np.nan, "min": np.nan, "max": np.nan,
            "max_drawdown": np.nan, "VaR_5%": np.nan, "CVaR_5%": np.nan
        })
    mean_m = r.mean()
    std_m = r.std(ddof=1)
    var5 = np.quantile(r, 0.05)
    cvar5 = r[r <= var5].mean() if (r <= var5).any() else np.nan
    return pd.Series({
        "n": len(r),
        "mean_m": mean_m,
        "median_m": r.median(),
        "std_m": std_m,
        "ann_mean": 12 * mean_m,
        "ann_vol": np.sqrt(12) * std_m,
        "sharpe_0rf": (mean_m / std_m) * np.sqrt(12) if std_m != 0 else np.nan,
        "win_rate": (r > 0).mean(),
        "min": r.min(),
        "max": r.max(),
        "max_drawdown": max_drawdown(r),
        "VaR_5%": var5,
        "CVaR_5%": cvar5
    })

def rolling_sharpe(r, window=12):
    r = pd.Series(r).dropna()
    m = r.rolling(window).mean()
    s = r.rolling(window).std(ddof=1)
    return (m / s) * np.sqrt(12)

def cross_sectional_long_short(signal, rets_df):
    """Rank by signal -> long winner, short loser, return = r_long - r_short."""
    valid = signal.dropna(how="all")
    if valid.empty:
        return pd.DataFrame(columns=["long", "short", "ls_ret"])

    long_name = valid.idxmax(axis=1, skipna=True)
    short_name = valid.idxmin(axis=1, skipna=True)

    aligned_rets = rets_df.loc[valid.index]
    col_to_pos = {c: i for i, c in enumerate(aligned_rets.columns)}
    long_pos = long_name.map(col_to_pos).to_numpy()
    short_pos = short_name.map(col_to_pos).to_numpy()

    R = aligned_rets.to_numpy()
    row = np.arange(len(aligned_rets))
    ls_ret = R[row, long_pos] - R[row, short_pos]

    return pd.DataFrame({"long": long_name, "short": short_name, "ls_ret": ls_ret}, index=valid.index)

# =========================
# PRINT INDEX STATS
# =========================
print("\nIndex stats (monthly returns):")
for c in rets.columns:
    print(c)
    print(stats(rets[c]).to_string())
    print()

# =========================
# STRATEGIES
# =========================
results = {}

if DO_1M:
    signal_1m = rets.shift(1)
    ls_1m = cross_sectional_long_short(signal_1m, rets)
    results["LS_1M"] = ls_1m
    print("\nCross-sectional L/S (1M signal: last month winner - loser):")
    print(stats(ls_1m["ls_ret"]).to_string())

mom12 = rets.rolling(12).apply(cumret, raw=True)

if DO_12M:
    signal_12m = mom12.shift(1)
    ls_12m = cross_sectional_long_short(signal_12m, rets)
    results["LS_12M"] = ls_12m
    print("\nCross-sectional L/S (12M formation, no skip):")
    print(stats(ls_12m["ls_ret"]).to_string())

if DO_12M_1:
    signal_12m_1 = mom12.shift(2)
    ls_12m_1 = cross_sectional_long_short(signal_12m_1, rets)
    results["LS_12M_1"] = ls_12m_1
    print("\nCross-sectional L/S (12–1 formation, skip most recent month):")
    print(stats(ls_12m_1["ls_ret"]).to_string())

# =========================
# SERIES FOR PLOTTING (attach plot_index only at the end)
# =========================
series = rets.copy()
for k, v in results.items():
    series[k] = np.nan
    series.loc[v.index, k] = v["ls_ret"].values

series.index = plot_index  # plot x-axis only

# =========================
# CHART 1: Growth of 1 (Indices)
# =========================
growth_idx = (1 + series[["IND", "US", "JAPAN"]].fillna(0)).cumprod()

plt.figure()
for c in ["IND", "US", "JAPAN"]:
    plt.plot(growth_idx.index, growth_idx[c], label=c)
plt.title("Growth of 1: Indices (Price Returns)")
plt.xlabel("Date")
plt.ylabel("Cumulative Growth")
plt.legend()
plt.tight_layout()
plt.savefig("chart_1_growth_indices.png", dpi=200)

# =========================
# CHART 2: Growth of 1 (Strategies)
# =========================
strategy_cols = [c for c in ["LS_1M", "LS_12M", "LS_12M_1"] if c in series.columns]

if strategy_cols:
    growth_strat = (1 + series[strategy_cols].fillna(0)).cumprod()
    plt.figure()
    for c in strategy_cols:
        plt.plot(growth_strat.index, growth_strat[c], label=c)
    plt.title("Growth of 1: Long-Short Momentum Strategies")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chart_2_growth_strategies.png", dpi=200)

# =========================
# CHART 3: Drawdowns (Strategies)
# =========================
if strategy_cols:
    plt.figure()
    for c in strategy_cols:
        dd = drawdown_series(series[c])
        plt.plot(dd.index, dd, label=c)
    plt.title("Drawdowns: Long-Short Momentum Strategies")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chart_3_drawdowns_strategies.png", dpi=200)

# =========================
# CHART 4: Rolling 12M Sharpe (Strategies)
# =========================
if strategy_cols:
    plt.figure()
    for c in strategy_cols:
        rs = rolling_sharpe(series[c], window=ROLLING_WINDOW)
        plt.plot(rs.index, rs, label=c)
    plt.title(f"Rolling {ROLLING_WINDOW}-Month Sharpe (0% RF): Strategies")
    plt.xlabel("Date")
    plt.ylabel("Rolling Sharpe")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chart_4_rolling_sharpe.png", dpi=200)

# =========================
# EXPORT (data + long/short labels)
# =========================
export = series.copy()

for k, v in results.items():
    tmp = pd.DataFrame(index=series.index, columns=[f"{k}_long", f"{k}_short"], dtype=object)
    tmp.iloc[v.index] = np.column_stack([v["long"].astype(str).values, v["short"].astype(str).values])
    export = pd.concat([export, tmp], axis=1)

out_path = "international_momentum_output_with_charts_data.xlsx"
export.to_excel(out_path, index_label="Date")

print("\nSaved charts:")
print(" - chart_1_growth_indices.png")
print(" - chart_2_growth_strategies.png")
print(" - chart_3_drawdowns_strategies.png")
print(" - chart_4_rolling_sharpe.png")
print(f"Saved output data -> {out_path}")

# =========================
# REGRESSIONS (HAC / Newey–West) + CHART 5
# =========================
# Requires: python -m pip install statsmodels
import statsmodels.api as sm

def nw_ols(y, X=None, lags=3):
    """
    OLS with Newey–West (HAC) standard errors.
    Returns fitted model with coef/t/p.
    """
    y = pd.Series(y).dropna()
    if X is None:
        X = pd.DataFrame(index=y.index)
    else:
        X = X.loc[y.index].copy()
    X = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

def print_reg_table(model, title):
    tbl = pd.DataFrame({"coef": model.params, "t": model.tvalues, "p": model.pvalues})
    print(f"\n{title}")
    print(tbl.to_string())

def run_regs(strat_name):
    if strat_name not in series.columns:
        return None, None, None

    y = series[strat_name].dropna()

    # A) Alpha-only
    mA = nw_ols(y, X=None, lags=NW_LAGS)
    print_reg_table(mA, f"Regression A (HAC): {strat_name} = alpha")

    # B) Market-adjusted (US proxy)
    mB = nw_ols(y, X=series[["US"]], lags=NW_LAGS)
    print_reg_table(mB, f"Regression B (HAC): {strat_name} = alpha + beta*US")

    # C) Regime: High-vol dummy based on US 12M rolling vol
    us_vol12 = series["US"].rolling(12).std()
    high_vol = (us_vol12.shift(1) > us_vol12.shift(1).median()).astype(int)
    mC = nw_ols(y, X=pd.DataFrame({"HighVol_US": high_vol}), lags=NW_LAGS)
    print_reg_table(mC, f"Regression C (HAC): {strat_name} = alpha + gamma*HighVol_US")

    return mA, mB, mC

# Run regressions for all strategies you computed
for strat in ["LS_1M", "LS_12M", "LS_12M_1"]:
    run_regs(strat)

# =========================
# CHART 5: Scatter (Strategy vs US) with regression line
# =========================
if MAIN_STRAT in series.columns:
    tmp = pd.concat([series["US"], series[MAIN_STRAT]], axis=1).dropna()
    tmp.columns = ["US", MAIN_STRAT]

    x = tmp["US"].values
    y = tmp[MAIN_STRAT].values

    # Fit line y = a + b*x
    b, a = np.polyfit(x, y, 1)

    plt.figure()
    plt.scatter(x, y, label="Monthly points")
    xx = np.linspace(x.min(), x.max(), 100)
    plt.plot(xx, a + b * xx, label="Fitted line")
    plt.title(f"Chart 5: {MAIN_STRAT} vs US monthly return (with fitted line)")
    plt.xlabel("US monthly return")
    plt.ylabel(f"{MAIN_STRAT} monthly return")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chart_5_scatter_strategy_vs_us.png", dpi=200)

    print(" - chart_5_scatter_strategy_vs_us.png")

plt.show()
