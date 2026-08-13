"""
load_data.py
Parse Robert Shiller's ie_data.xls (downloaded from econ.yale.edu / shillerdata.com)
and build clean monthly REAL total-return series for US stocks and 10y Treasury bonds.

Output: shiller_monthly.csv with one row per month (Jan 1871 .. Sep 2023) containing:
  year, month, dec_date, P, D, CPI, GS10, CAPE,
  stock_ret_real  (monthly real total return, dividends reinvested)
  bond_ret_real   (monthly real total return, 10y constant-maturity approximation)

All returns are simple monthly returns in REAL (CPI-deflated) terms.

Bond series is an APPROXIMATION built from the GS10 long yield (see build_bond_returns).
We validate it against Shiller's own "Real Total Bond Returns" column.
"""
import numpy as np
import pandas as pd

RAW = "ie_data.xls"
OUT = "shiller_monthly.csv"

# Column indices in the 'Data' sheet (0-based), header spans rows 4-7, data starts row 8.
COL = {
    "date": 0,      # fractional year e.g. 1871.01 => Jan 1871
    "P": 1,         # S&P Composite price
    "D": 2,         # dividend (annualized $)
    "E": 3,         # earnings
    "CPI": 4,       # CPI-U
    "GS10": 6,      # 10-year interest rate (long govt bond yield), percent
    "CAPE": 12,     # cyclically adjusted P/E (P/E10)
    "bond_tr_nom": 17,  # Shiller monthly nominal total bond return index (for validation)
    "bond_tr_real": 18, # Shiller real total bond return index (for validation)
}


def load_raw():
    raw = pd.read_excel(RAW, sheet_name="Data", header=None)
    # data rows: from row 8 until the date column stops being numeric
    rows = []
    for i in range(8, raw.shape[0]):
        d = raw.iloc[i, COL["date"]]
        if not isinstance(d, (int, float)) or pd.isna(d):
            break
        rec = {k: raw.iloc[i, c] for k, c in COL.items()}
        rows.append(rec)
    df = pd.DataFrame(rows)
    # decode fractional date -> year, month
    # Shiller encodes month as .01=Jan ... .10=Oct .11=Nov .12=Dec (two-decimal month)
    df["year"] = np.floor(df["date"] + 1e-6).astype(int)
    frac = df["date"] - df["year"]
    df["month"] = np.round(frac * 100).astype(int)
    df["dec_date"] = df["year"] + (df["month"] - 0.5) / 12.0
    for c in ["P", "D", "CPI", "GS10", "CAPE", "bond_tr_nom", "bond_tr_real"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_stock_returns(df):
    """Monthly nominal total return = (P_{t+1} + D_t/12) / P_t - 1.
    D is the annualized dividend, so one month's cash dividend = D/12.
    Then deflate to real using CPI."""
    P = df["P"].values
    D = df["D"].values
    CPI = df["CPI"].values
    n = len(df)
    nom = np.full(n, np.nan)
    for t in range(n - 1):
        if P[t] > 0 and not np.isnan(P[t + 1]):
            div = (D[t] / 12.0) if not np.isnan(D[t]) else 0.0
            nom[t] = (P[t + 1] + div) / P[t] - 1.0
    # real: multiply gross nominal by CPI_t/CPI_{t+1}
    real = np.full(n, np.nan)
    for t in range(n - 1):
        if not np.isnan(nom[t]) and CPI[t + 1] > 0:
            real[t] = (1.0 + nom[t]) * CPI[t] / CPI[t + 1] - 1.0
    df["stock_ret_nom"] = nom
    df["stock_ret_real"] = real
    return df


def price_par_bond(coupon, ytm, n_years=10, freq=1):
    """Price (per 1 face) of a bond with annual coupon rate `coupon`, valued at yield `ytm`,
    `n_years` to maturity, `freq` payments/yr. Both rates are annual decimals."""
    m = int(n_years * freq)
    c = coupon / freq
    y = ytm / freq
    if y == 0:
        return c * m + 1.0
    disc = (1 + y) ** (-np.arange(1, m + 1))
    return c * disc.sum() + disc[-1]


def build_bond_returns(df, n_years=10, freq=1):
    """Constant-maturity 10y Treasury total return approximation.
    Each month you hold a par 10y bond (coupon = start-of-month yield y_t, price=1).
    One month later the yield is y_{t+1}; reprice the SAME bond assuming ~10y still remain
    (constant-maturity approx), and add one month of coupon accrual (y_t/12).
      r_nom(t) = y_t/12 + (P(coupon=y_t, ytm=y_{t+1}) - 1)
    Then deflate to real via CPI. This is the standard Shiller-data bond approximation."""
    y = df["GS10"].values / 100.0  # decimals
    CPI = df["CPI"].values
    n = len(df)
    nom = np.full(n, np.nan)
    for t in range(n - 1):
        yt, yt1 = y[t], y[t + 1]
        if np.isnan(yt) or np.isnan(yt1):
            continue
        p_new = price_par_bond(yt, yt1, n_years=n_years, freq=freq)
        nom[t] = yt / 12.0 + (p_new - 1.0)
    real = np.full(n, np.nan)
    for t in range(n - 1):
        if not np.isnan(nom[t]) and CPI[t + 1] > 0:
            real[t] = (1.0 + nom[t]) * CPI[t] / CPI[t + 1] - 1.0
    df["bond_ret_nom"] = nom
    df["bond_ret_real"] = real
    return df


def validate(df):
    """Compare our bond approximation to Shiller's own bond TR index (cols 17/18)."""
    # Shiller's real bond TR is an index level; convert to monthly returns.
    idx = df["bond_tr_real"].values
    sh_ret = np.full(len(df), np.nan)
    for t in range(len(df) - 1):
        if idx[t] > 0 and not np.isnan(idx[t + 1]):
            sh_ret[t] = idx[t + 1] / idx[t] - 1.0
    ours = df["bond_ret_real"].values
    mask = ~np.isnan(sh_ret) & ~np.isnan(ours)
    if mask.sum() > 12:
        corr = np.corrcoef(sh_ret[mask], ours[mask])[0, 1]
        # cumulative real growth over the overlap
        cum_ours = np.prod(1 + ours[mask])
        cum_sh = np.prod(1 + sh_ret[mask])
        ann_ours = cum_ours ** (12 / mask.sum()) - 1
        ann_sh = cum_sh ** (12 / mask.sum()) - 1
        print(f"[bond validation] overlap months={mask.sum()}  monthly-return corr={corr:.3f}")
        print(f"[bond validation] annualized REAL return  ours={ann_ours*100:.2f}%  Shiller={ann_sh*100:.2f}%")
    else:
        print("[bond validation] insufficient overlap with Shiller bond column")

    # stock sanity: annualized real total return over full sample
    s = df["stock_ret_real"].dropna().values
    ann_s = np.prod(1 + s) ** (12 / len(s)) - 1
    print(f"[stock check] full-sample annualized REAL total return = {ann_s*100:.2f}%  (expect ~6.5-7%)")
    b = df["bond_ret_real"].dropna().values
    ann_b = np.prod(1 + b) ** (12 / len(b)) - 1
    print(f"[bond check]  full-sample annualized REAL total return = {ann_b*100:.2f}%  (expect ~2-3%)")


def main():
    df = load_raw()
    df = build_stock_returns(df)
    df = build_bond_returns(df)
    print(f"Loaded {len(df)} months: {df['year'].iloc[0]}.{df['month'].iloc[0]:02d} "
          f".. {df['year'].iloc[-1]}.{df['month'].iloc[-1]:02d}")
    validate(df)
    keep = ["year", "month", "dec_date", "P", "D", "CPI", "GS10", "CAPE",
            "stock_ret_nom", "stock_ret_real", "bond_ret_nom", "bond_ret_real"]
    df[keep].to_csv(OUT, index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
