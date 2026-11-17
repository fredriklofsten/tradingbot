"""
AI-tradingbot v2 – två låtsasportföljer sida vid sida.

  AI-portföljen:   äger alla aktier, men lägger MER pengar i de aktier AI-modellen
                   (och nyheterna) tror mest på, och mindre i de den tror minst på.
  Enkla portföljen: lika mycket i alla aktier. Ingen AI.
  Båda:            halverar insatsen i en marknad (USA/Sverige) när marknaden ligger
                   under sitt 200-dagars snitt. Ombalanserar en gång i månaden.

Varför: i testerna 2003–idag gav spridning + marknadsfilter bäst riskjusterad
avkastning. AI-delen hade nästan ingen förmåga att välja vinnare. Nu får
verkligheten avgöra om AI:n tillför något.

KOMMANDON
    python tradingbot.py backtest   # historiktest (några minuter)
    python tradingbot.py papper     # en handelsdag med låtsaspengar
    python tradingbot.py rapport    # bygger rapport.html + docs/ (webbappen)
    python tradingbot.py allt       # papper + rapport (körs varje vardag)
"""
import base64
import io
import json
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ---------------- Inställningar ----------------
US = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "JNJ", "XOM", "PG",
      "KO", "WMT", "V", "UNH", "HD"]
SE = ["VOLV-B.ST", "ERIC-B.ST", "INVE-B.ST", "ATCO-A.ST", "ABB.ST", "AZN.ST", "SEB-A.ST",
      "SHB-A.ST", "HM-B.ST", "SAND.ST", "ASSA-B.ST", "ESSITY-B.ST", "HEXA-B.ST", "SKF-B.ST",
      "TELIA.ST", "EVO.ST", "SWED-A.ST", "ALFA.ST"]
TICKERS = US + SE
MARKET_US, MARKET_SE = "SPY", "^OMX"
START = "2000-01-01"
TEST_START = "2003-01-01"   # AI:n behöver några år att träna på innan första testet
STEP_MONTHS = 12            # AI:n tränas om en gång per år i historiktestet
HORIZON = 20            # förutsäg ca 1 månad framåt
COST = 0.0015           # courtage + spread per affär
FX_FEE = 0.0025         # valutaväxling hos Avanza/Nordnet för amerikanska aktier, per affär


def cost_of(t: str) -> float:
    """Kostnad per affär som andel av beloppet: svenska 0,15 %, amerikanska 0,40 %."""
    return COST + (0.0 if t.endswith(".ST") else FX_FEE)
REBALANCE_DAYS = 20     # ombalansera ungefär en gång i månaden
FILTER_FACTOR = 0.5     # insats när marknaden är under 200-dagars snitt
TILT = 1.0              # hur mycket AI:n får flytta vikterna (0 = ingen AI)
NEWS_WEIGHT = 0.05
NEWS_DAYS = 7
START_CASH = 100_000
MIN_TRADE = 0.005       # handla inte om skillnaden är under 0,5 % av portföljen
CLAUDE_MODEL = "claude-haiku-5-5"
VERSION = 7

DIR = os.path.dirname(os.path.abspath(__file__))
F_PORTFOLIO = os.path.join(DIR, "papper_portfolj.json")
F_TRADES = os.path.join(DIR, "papper_affarer.csv")
F_HISTORY = os.path.join(DIR, "papper_historik.csv")
F_PRED = os.path.join(DIR, "prediktioner.csv")
F_BACKTEST = os.path.join(DIR, "backtest.json")
F_REPORT = os.path.join(DIR, "rapport.html")
PORTS = {"ai": "AI-portföljen", "enkel": "Enkla portföljen", "risk": "Risk-portföljen",
         "hold": "Gör ingenting"}
INDEX_FUND = "ACWI"   # global indexfond (MSCI All Country World) som måttstock
F_NEWS = os.path.join(DIR, "nyheter_arkiv.csv")
F_SHORT = os.path.join(DIR, "blankning.csv")
MACRO = ["vix", "vix_chg20", "ranta10", "ranta_chg60", "usdsek_chg60"]

MODEL_PARAMS = dict(max_iter=300, learning_rate=0.03, max_depth=4,
                    min_samples_leaf=200, l2_regularization=1.0, random_state=42)


# ---------------- Data ----------------
_cache: dict = {}


def load_data(ticker: str) -> pd.DataFrame:
    if ticker in _cache:
        return _cache[ticker]
    if ticker == MARKET_SE:
        _cache[ticker] = _omx_spliced()
        return _cache[ticker]
    import yfinance as yf
    df = yf.download(ticker, start=START, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    # Använd bara färdiga dagar: släng dagens stapel om marknaden inte har stängt än
    now = pd.Timestamp.now(tz="Europe/Stockholm")
    closes_at = (17, 45) if ticker.endswith(".ST") or ticker == MARKET_SE else (22, 15)
    if len(df) and df.index[-1].date() == now.date() and (now.hour, now.minute) < closes_at:
        df = df.iloc[:-1]
    _cache[ticker] = df
    return df


def _omx_spliced() -> pd.DataFrame:
    """Stockholmsbörsens index (^OMX) finns bara från nov 2008. Före det används ett eget
    likaviktat index av de svenska aktierna, skarvat så att nivåerna möts."""
    import yfinance as yf
    df = yf.download(MARKET_SE, start=START, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    omx = df["Close"].dropna()
    rets = pd.DataFrame({t: load_data(t)["Close"].pct_change() for t in SE}).mean(axis=1, skipna=True)
    proxy = (1 + rets.fillna(0)).cumprod()
    first = omx.index[0]
    proxy = proxy[proxy.index < first] * (omx.iloc[0] / proxy[proxy.index <= first].iloc[-1])
    close = pd.concat([proxy, omx])
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 0.0})


def market_of(t: str) -> str:
    return MARKET_SE if t.endswith(".ST") else MARKET_US


def close_on(ticker: str, day) -> float:
    """Senaste stängningskurs på eller före en viss dag."""
    c = load_data(ticker)["Close"].dropna()
    return float(c[c.index <= pd.Timestamp(day)].iloc[-1])


def to_sek(t: str, price: float, fx: float) -> float:
    return price if t.endswith(".ST") else price * fx


# ---------------- Egenskaper ----------------
def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n).mean()
    return 100 - 100 / (1 + up / down)


def make_features(df: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    c, v = df["Close"], df["Volume"]
    f = pd.DataFrame(index=df.index)
    for n in (1, 5, 10, 20, 60, 120):
        f[f"avk_{n}d"] = c.pct_change(n)
    for n in (10, 20, 50, 200):
        f[f"avst_ma{n}"] = c / c.rolling(n).mean() - 1
    f["vol_20"] = c.pct_change().rolling(20).std()
    f["vol_60"] = c.pct_change().rolling(60).std()
    f["vol_kvot"] = f["vol_20"] / f["vol_60"]
    f["rsi_14"] = rsi(c)
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - c.shift()).abs(),
                    (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c
    f["volym_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    f["fran_arshogsta"] = c / c.rolling(252).max() - 1
    m = market.reindex(df.index).ffill()
    f["marknad_avk_20d"] = m.pct_change(20)
    f["marknad_avst_ma200"] = m / m.rolling(200).mean() - 1
    f["relativ_styrka_60d"] = c.pct_change(60) - m.pct_change(60)
    return f


RANK_COLS = ["avk_20d", "avk_60d", "avk_120d", "vol_60", "relativ_styrka_60d", "fran_arshogsta"]


def fx_series() -> pd.Series:
    return load_data("USDSEK=X")["Close"]


def build_panel() -> tuple[pd.DataFrame, list]:
    frames = []
    for t in TICKERS:
        try:
            df = load_data(t)
            mkt = load_data(market_of(t))["Close"]
        except Exception as e:
            print(f"   {t}: kunde inte hämta ({e})")
            continue
        if len(df) < 1000:
            continue
        f = make_features(df, mkt)
        # Avkastning räknas i kronor (amerikanska aktier räknas om med dollarkursen)
        px = df["Close"] if t.endswith(".ST") else df["Close"] * fx_series().reindex(df.index).ffill().bfill()
        f["fwd"] = px.shift(-HORIZON) / px - 1
        f["ret1"] = px.pct_change().shift(-1)
        f["pris"] = df["Close"]
        f["aktie"] = t
        f["se"] = int(t.endswith(".ST"))
        frames.append(f)
    p = pd.concat(frames)
    p.index.name = "Date"
    p = p.reset_index().dropna(subset=["avst_ma200", "fran_arshogsta"])
    for c in RANK_COLS:  # jämför bara med aktier på samma börs (USA stänger senare än Stockholm)
        p[c + "_rank"] = p.groupby(["Date", "se"])[c].rank(pct=True)
    # Mål: slår aktien mitten-aktien den kommande månaden?
    med = p.groupby("Date")["fwd"].transform("median")
    p["y"] = (p["fwd"] > med).astype(float).where(p["fwd"].notna())
    feats = [c for c in p.columns if c not in ("Date", "fwd", "ret1", "pris", "aktie", "y")]
    # Makrodata (används av riskmodellen)
    vix, tnx = load_data("^VIX")["Close"], load_data("^TNX")["Close"]
    fx = load_data("USDSEK=X")["Close"]
    macro = pd.DataFrame({"vix": vix, "vix_chg20": vix.pct_change(20), "ranta10": tnx,
                          "ranta_chg60": tnx.diff(60), "usdsek_chg60": fx.pct_change(60)}).ffill().shift(1)
    p = p.merge(macro, left_on="Date", right_index=True, how="left")
    # Mål för riskmodellen: hur mycket aktien svänger kommande månad (årstakt)
    p["fwd_vol"] = p.groupby("aktie")["ret1"].transform(
        lambda r: r.rolling(20).std().shift(-19)) * np.sqrt(252)
    p["log_fwd_vol"] = np.log(p["fwd_vol"] + 1e-4)
    return p, feats


def risk_feats(feats):
    return feats + MACRO


def new_risk_model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4,
                                         min_samples_leaf=200, random_state=42)


def new_model():
    return HistGradientBoostingClassifier(**MODEL_PARAMS)


# ---------------- Blankning (Finansinspektionen) ----------------
BLANK_LIMIT = 2.0       # aktier där minst 2 % av aktierna är blankade ...
BLANK_FACTOR = 0.3      # ... får bara 30 % av sin vanliga vikt
BLANK_NAMN = {"VOLV-B.ST": r"^(?:Aktiebolaget|AB) Volvo", "ERIC-B.ST": "Ericsson", "INVE-B.ST": r"^Investor",
              "ATCO-A.ST": "Atlas Copco", "ABB.ST": r"^ABB", "AZN.ST": "ASTRAZENECA|AstraZeneca",
              "SEB-A.ST": "Skandinaviska Enskilda", "SHB-A.ST": "Handelsbanken", "HM-B.ST": "Hennes",
              "SAND.ST": "Sandvik", "ASSA-B.ST": "ASSA ABLOY", "ESSITY-B.ST": "Essity",
              "HEXA-B.ST": r"^Hexagon", "SKF-B.ST": "SKF", "TELIA.ST": "Telia",
              "EVO.ST": r"^Evolution (?:AB|Gaming)", "SWED-A.ST": "Swedbank", "ALFA.ST": "Alfa Laval"}


def load_short_interest() -> pd.DataFrame:
    """Summan av alla rapporterade blankningspositioner (≥0,5 %) per svensk aktie och dag.
    Byggs från Finansinspektionens historikfil. Tom tabell om FI inte går att nå."""
    import requests
    import time
    raw = None
    for attempt in range(8):  # FI:s server svarar ibland 404 – försök igen
        try:
            r = requests.get("https://www.fi.se/BlankningsRegister/GetHistFile", timeout=180,
                             headers={"User-Agent": "Mozilla/5.0 tradingbot"})
            if "opendocument" in r.headers.get("content-type", ""):
                raw = pd.read_excel(io.BytesIO(r.content), engine="odf", header=None, skiprows=6).iloc[:, :6]
                break
        except Exception:
            pass
        time.sleep(4)
    if raw is None:
        if os.path.exists(F_SHORT):
            print("   Blankningsregistret svarade inte – använder senast sparade kopia")
            return pd.read_csv(F_SHORT, index_col=0, parse_dates=True)
        print("   Blankningsregistret gick inte att hämta – regeln används inte idag")
        return pd.DataFrame()
    raw.columns = ["holder", "emittent", "isin", "pct", "datum", "kommentar"]
    raw["datum"] = pd.to_datetime(raw["datum"], errors="coerce")
    raw["pct"] = pd.to_numeric(raw["pct"].astype(str).str.replace("<0,5", "0").str.replace(",", "."),
                               errors="coerce")
    raw = raw.dropna(subset=["pct", "datum", "emittent"])
    days = pd.bdate_range("2012-01-01", pd.Timestamp.today())
    out = {}
    for t, rx in BLANK_NAMN.items():
        d = raw[raw["emittent"].str.contains(rx, regex=True, na=False)]
        tot = pd.Series(0.0, index=days)
        for _, g in d.groupby("holder"):
            s = g.sort_values("datum").groupby("datum")["pct"].last()
            s = s.where(s >= 0.5, 0.0)
            tot += s.reindex(days.union(s.index)).ffill().fillna(0).reindex(days)
        out[t] = tot
    sh = pd.DataFrame(out).shift(1)  # publiceras dagen efter
    sh.loc["2016-01-01":].round(2).to_csv(F_SHORT)
    return sh


def add_short_interest(p: pd.DataFrame) -> pd.DataFrame:
    sh = load_short_interest()
    if sh.empty:
        p["blankning"] = np.nan
        return p
    long = sh.stack().rename("blankning").reset_index()
    long.columns = ["Date", "aktie", "blankning"]
    return p.merge(long, on=["Date", "aktie"], how="left")


# ---------------- Portföljvikter ----------------
BASE_CFG = {"tilt": TILT, "filter_factor": FILTER_FACTOR, "reb_days": REBALANCE_DAYS,
            "blank_limit": BLANK_LIMIT, "blank_factor": BLANK_FACTOR}


def target_weights(scores: pd.Series, below: dict, mode: str, pred_vol=None, blank=None,
                   cfg=None) -> pd.Series:
    """mode: 'ai' (mer i aktier AI:n tror på), 'enkel' (lika), 'risk' (mindre i stormiga aktier),
    'ai_risk' (båda). below: {'us': bool, 'se': bool} – marknaden under sitt 200-dagars snitt.
    blank: blankning i procent per aktie – mycket blankade aktier får lägre vikt."""
    cfg = {**BASE_CFG, **(cfg or {})}
    w = pd.Series(1.0, index=scores.index)
    if mode in ("ai", "ai_risk"):
        w *= (1 + (scores.rank(pct=True) - 0.5) * 2 * cfg["tilt"]).clip(lower=0)
    if mode in ("risk", "ai_risk") and pred_vol is not None:
        w /= pred_vol.reindex(scores.index).fillna(pred_vol.median()).clip(lower=0.05)
    if blank is not None:
        heavy = (blank.reindex(scores.index) >= cfg["blank_limit"]).fillna(False)
        w[heavy] *= cfg["blank_factor"]
    w = w / w.sum()
    for t in w.index:
        if below["se" if t.endswith(".ST") else "us"]:
            w[t] *= cfg["filter_factor"]
    return w


# ---------------- Backtest ----------------
def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    years = max((r.index[-1] - r.index[0]).days / 365.25, 1 / 365)
    per_year = len(r) / years
    return {"arlig": float(eq.iloc[-1] ** (1 / years) - 1),
            "sharpe": float(r.mean() / r.std() * np.sqrt(per_year)) if r.std() > 0 else 0.0,
            "max_ras": float((eq / eq.cummax() - 1).min())}


def walk_forward(p: pd.DataFrame, feats: list, model_fn=None, risk=True, verbose=False, start=TEST_START):
    """Tränar om varje år (STEP_MONTHS) och förutsäger perioden efter, som modellen aldrig sett."""
    model_fn = model_fn or new_model
    p = p.copy()
    p["prob"], p["pred_vol"] = np.nan, np.nan
    rf = risk_feats(feats)
    learning = []
    start = pd.Timestamp(start)
    while start < p["Date"].max():
        end = start + pd.DateOffset(months=STEP_MONTHS)
        tr = p[p["Date"] < start - pd.Timedelta(days=35)].dropna(subset=["y"])
        te = (p["Date"] >= start) & (p["Date"] < end)
        if te.any():
            m = model_fn().fit(tr[feats], tr["y"])
            p.loc[te, "prob"] = m.predict_proba(p.loc[te, feats])[:, 1]
            if risk:
                trv = p[p["Date"] < start - pd.Timedelta(days=35)].dropna(subset=["log_fwd_vol"])
                mv = new_risk_model().fit(trv[rf], trv["log_fwd_vol"])
                p.loc[te, "pred_vol"] = np.exp(mv.predict(p.loc[te, rf]))
            sc = p.loc[te].dropna(subset=["y"])
            if len(sc) > 200 and sc["y"].nunique() == 2:
                e = {"period": start.strftime("%Y-%m"), "auc": float(roc_auc_score(sc["y"], sc["prob"])),
                     "traningsrader": len(tr)}
                if risk:
                    sv = p.loc[te].dropna(subset=["fwd_vol"])
                    e["risk_fel_ai"] = float((sv["pred_vol"] - sv["fwd_vol"]).abs().mean())
                    e["risk_fel_enkel"] = float((sv["vol_20"] * np.sqrt(252) - sv["fwd_vol"]).abs().mean())
                learning.append(e)
                if verbose:
                    print(f"   {start:%Y-%m}: AUC {e['auc']:.3f}")
        start = end
    return p, learning


def simulate(q: pd.DataFrame, mode: str, cfg=None, use_filter=True, score_col="prob",
             start=None) -> pd.Series:
    """Daglig avkastning för en portfölj, med courtage. Portföljer som behöver AI:ns prognos
    startar först när prognoserna finns (2003); regelportföljer kan starta tidigare."""
    cfg = {**BASE_CFG, **(cfg or {})}
    needs_model = mode in ("ai", "ai_risk", "risk") or score_col != "prob"
    if needs_model:
        mask = q[score_col].notna()
        if mode in ("risk", "ai_risk"):
            mask &= q["pred_vol"].notna()
        first = q.loc[mask, "Date"].min()
    else:
        first = q.loc[q["avst_ma200"].notna(), "Date"].min()
    q = q[q["Date"] >= max(pd.Timestamp(start or first), first)]
    R = q.pivot(index="Date", columns="aktie", values="ret1").fillna(0)
    # Fyll i helgdagar (när bara den ena börsen är öppen) med senaste kända värde
    piv = lambda c: q.pivot(index="Date", columns="aktie", values=c).ffill(limit=5)
    P, reg, PV = piv(score_col), piv("marknad_avst_ma200"), piv("pred_vol")
    BL = piv("blankning") if "blankning" in q else None
    AV = piv("avst_ma200").notna()
    us = [c for c in R.columns if not c.endswith(".ST")]
    se = [c for c in R.columns if c.endswith(".ST")]
    W = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    w, last_below, since = None, None, 0
    for i, d in enumerate(R.index):
        r_d = reg.loc[d]
        below = {"us": bool((r_d[us] < 0).mean() > 0.5), "se": bool((r_d[se] < 0).mean() > 0.5)}
        if not use_filter:
            below = {"us": False, "se": False}
        since += 1
        if w is None or since >= cfg["reb_days"] or below != last_below:
            if needs_model:
                s = P.loc[d].dropna()
            else:
                avail = AV.loc[d]
                s = pd.Series(0.5, index=avail[avail].index)
            if len(s) > 10:
                w = target_weights(s, below, mode, PV.loc[d], BL.loc[d] if BL is not None else None,
                                   cfg).reindex(R.columns).fillna(0)
                last_below, since = below, 0
        if w is not None:
            W.loc[d] = w
    W = W.shift(1).fillna(0.0)  # beslut vid stängning dag d, affären görs till stängning dag d+1
    gross = (W * R).sum(axis=1)
    costs = pd.Series({c: cost_of(c) for c in R.columns})
    turn_cost = (W.diff().abs().fillna(W.abs()) * costs).sum(axis=1)
    return gross - turn_cost


def buy_and_hold(q: pd.DataFrame, start=None) -> pd.Series:
    """Köper alla aktier lika mycket första dagen och rör dem sedan aldrig."""
    R = q.pivot(index="Date", columns="aktie", values="ret1")
    AV = q.pivot(index="Date", columns="aktie", values="avst_ma200").notna()
    d0 = R.index[R.index >= pd.Timestamp(start)][0] if start else AV.index[AV.sum(axis=1) > 10][0]
    R, names = R.loc[d0:], AV.loc[d0][AV.loc[d0]].index
    growth = (1 + R[names].fillna(0)).cumprod()
    value = growth.mean(axis=1)
    r = value.pct_change()
    r.iloc[0] = value.iloc[0] - 1
    costs = np.mean([cost_of(t) for t in names])
    r.iloc[0] -= costs  # köpet första dagen
    return r


CRISES = {"IT-kraschen 2001–2002": ("2001-01-02", "2002-10-09"),  # testet kan starta först 2001
          "Finanskrisen 2007–2009": ("2007-10-09", "2009-03-09"),
          "Coronakraschen 2020": ("2020-02-19", "2020-03-23"),
          "Räntechocken 2022": ("2022-01-03", "2022-10-12")}


def cmd_backtest():
    print("Bygger data för alla aktier (från år 2000)...")
    p, feats = build_panel()
    p = add_short_interest(p)
    q, learning = walk_forward(p, feats, verbose=True)
    long = {"Enkla portföljen": simulate(q, "enkel"), "Gör ingenting": buy_and_hold(q)}
    spy = (load_data("SPY")["Close"] * fx_series().reindex(load_data("SPY").index).ffill().bfill()).pct_change()
    long["USA-börsen (SPY)"] = spy.shift(-1).reindex(q["Date"].drop_duplicates().sort_values()).fillna(0)
    res = {"AI-portföljen": simulate(q, "ai"), "Enkla portföljen": long["Enkla portföljen"],
           "Risk-portföljen": simulate(q, "risk"), "Gör ingenting": buy_and_hold(q, TEST_START),
           "USA-börsen (SPY)": long["USA-börsen (SPY)"]}
    out = {"version": VERSION, "skapad": datetime.now().isoformat(timespec="minutes"),
           "larande": learning, "strategier": {}, "kurvor": {}, "kriser": {}, "sakerhet": {},
           "period": f"{TEST_START[:4]}–idag"}
    for name, r in res.items():
        r = r.loc[TEST_START:]
        res[name] = r
        out["strategier"][name] = {"hela": stats(r), "fore_2022": stats(r.loc[:"2021-12-31"]),
                                   "fran_2022": stats(r.loc["2022-01-01":])}
        wk = (1 + r).cumprod().resample("W").last()
        out["kurvor"][name] = wk.round(4).tolist()
        out["kurvor"]["datum"] = [d.strftime("%Y-%m-%d") for d in wk.index]
        s = out["strategier"][name]["hela"]
        print(f"{name:22} {s['arlig']:+.1%}/år  Sharpe {s['sharpe']:.2f}  ras {s['max_ras']:.0%}")
    # Hur gick det i kriserna? (regelportföljerna går att testa redan från år 2000)
    series = {"AI-portföljen": res["AI-portföljen"], "Enkla portföljen": long["Enkla portföljen"],
              "Risk-portföljen": res["Risk-portföljen"], "Gör ingenting": long["Gör ingenting"],
              "USA-börsen (SPY)": long["USA-börsen (SPY)"]}
    for crisis, (a, b) in CRISES.items():
        row = {}
        for name, r in series.items():
            r = r.dropna()
            row[name] = (float((1 + r.loc[a:b]).prod() - 1) if len(r) and r.index[0] <= pd.Timestamp(a)
                         else None)
        out["kriser"][crisis] = row
        print(crisis, {k: (None if v is None else round(v, 3)) for k, v in row.items()})
    # Hur länge måste vi vänta innan en skillnad i verkligheten betyder något?
    for a_, b_ in (("AI-portföljen", "Gör ingenting"), ("Enkla portföljen", "Gör ingenting"),
                   ("AI-portföljen", "Enkla portföljen")):
        diff = (np.log1p(res[a_]) - np.log1p(res[b_])).dropna()  # log: jämför verklig tillväxt
        per_year = len(diff) / max((diff.index[-1] - diff.index[0]).days / 365.25, 1)
        d, te = float(diff.mean() * per_year), float(diff.std() * np.sqrt(per_year))
        out["sakerhet"][f"{a_}|{b_}"] = {"skillnad": d, "spridning": te,
                                          "ar": float((2 * te / abs(d)) ** 2) if abs(d) > 1e-6 else None}
    json.dump(out, open(F_BACKTEST, "w"), ensure_ascii=False, indent=1)
    try:
        lab = load_lab()
        last = pd.Timestamp(lab[-1]["datum"]) if lab else None
        if last is None or (pd.Timestamp.today() - last).days >= 6:  # en idé per vecka
            research_step(p, feats, q)
    except Exception as e:
        print(f"Forskningssteget misslyckades: {e}")


# ---------------- AI-forskaren ----------------
# Förregistrerade idéer, testas en i taget i den här ordningen. Listan är bestämd i förväg så att
# forskaren inte kan leta tills den råkar hitta något som ser bra ut.
DEV_START = TEST_START
DEV_END = "2023-12-31"      # idéer prövas på 2003–2023 ...
LOCKED_START = "2024-01-01"  # ... och bara de som klarar det kontrolleras på 2024–idag. Den perioden har
#                              redan setts när grundinställningarna valdes, så det riktiga provet för en idé
#                              som klarar båda är att köras med låtsaspengar.
MIN_GAIN = 0.03             # kräver minst så här mycket bättre Sharpe för att räknas som förbättring

IDEAS = [
    {"id": "tilt2", "titel": "Låt AI:n bestämma mer",
     "fraga": "Blir AI-portföljen bättre om AI:n får flytta pengarna dubbelt så mycket?",
     "portfolj": "ai", "cfg": {"tilt": 2.0}},
    {"id": "tilt05", "titel": "Låt AI:n bestämma mindre",
     "fraga": "Blir AI-portföljen bättre om AI:n bara får flytta pengarna hälften så mycket?",
     "portfolj": "ai", "cfg": {"tilt": 0.5}},
    {"id": "filter_ut", "titel": "Gå ur helt i nedgångar",
     "fraga": "Är det bättre att sälja allt (i stället för hälften) när börsen är under sitt 200-dagars snitt?",
     "portfolj": "enkel", "cfg": {"filter_factor": 0.0}},
    {"id": "reb40", "titel": "Byt innehav mer sällan",
     "fraga": "Blir det bättre att ombalansera varannan månad, så att courtaget blir lägre?",
     "portfolj": "enkel", "cfg": {"reb_days": 40}},
    {"id": "blank1", "titel": "Strängare blankningsregel",
     "fraga": "Blir det bättre att dra ner på aktier redan när 1 % är blankat (i stället för 2 %)?",
     "portfolj": "enkel", "cfg": {"blank_limit": 1.0}},
    {"id": "ai_risk", "titel": "Kombinera AI och risk",
     "fraga": "Blir det bättre att både följa AI:ns favoriter och undvika stormiga aktier?",
     "portfolj": "ai", "mode": "ai_risk", "cfg": {}},
    {"id": "momentum", "titel": "Momentum utan AI",
     "fraga": "Klarar en enkel regel (mer i aktier som gått bäst senaste halvåret) sig lika bra som AI:n?",
     "portfolj": "ai", "score": "avk_120d", "cfg": {}},
    {"id": "reb10", "titel": "Byt innehav oftare",
     "fraga": "Blir AI-portföljen bättre om den följer AI:ns senaste bedömning varannan vecka?",
     "portfolj": "ai", "cfg": {"reb_days": 10}},
    {"id": "logistisk", "titel": "Enklare AI-modell",
     "fraga": "Gör en mycket enklare modell (logistisk regression) samma nytta som den avancerade?",
     "portfolj": "ai", "modell": "logistisk", "cfg": {}},
    {"id": "bara_rank", "titel": "Bara jämförelser mellan aktier",
     "fraga": "Blir AI:n bättre om den bara får se hur aktierna står sig mot varandra, inte råa siffror?",
     "portfolj": "ai", "modell": "bara_rank", "cfg": {}},
    {"id": "filter_kvart", "titel": "Mildare marknadsfilter",
     "fraga": "Är det bättre att bara dra ner till 75 % (i stället för 50 %) när börsen är svag?",
     "portfolj": "enkel", "cfg": {"filter_factor": 0.75}},
    {"id": "blank_ut", "titel": "Undvik blankade aktier helt",
     "fraga": "Blir det bättre att inte äga mycket blankade aktier alls?",
     "portfolj": "enkel", "cfg": {"blank_factor": 0.0}},
]
F_LAB = os.path.join(DIR, "labbdagbok.json")


def load_lab():
    try:
        return json.load(open(F_LAB))
    except Exception:
        return []


def research_step(p, feats, q):
    lab = load_lab()
    done = {e["id"] for e in lab}
    idea = next((i for i in IDEAS if i["id"] not in done), None)
    if idea is None:
        print("AI-forskaren har testat alla idéer på listan.")
        return
    print(f"AI-forskaren testar: {idea['titel']}")
    base_mode = idea["portfolj"]
    qq, score = q, idea.get("score", "prob")
    if idea.get("modell") == "logistisk":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        fn = lambda: make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
        qq, _ = walk_forward(p, feats, model_fn=fn, risk=False)
        qq = qq.merge(q[["Date", "aktie", "pred_vol"]], on=["Date", "aktie"], how="left", suffixes=("_x", ""))
    elif idea.get("modell") == "bara_rank":
        fr = [f for f in feats if f.endswith("_rank")] + ["se"]
        qq, _ = walk_forward(p, fr, risk=False)
        qq = qq.merge(q[["Date", "aktie", "pred_vol"]], on=["Date", "aktie"], how="left", suffixes=("_x", ""))
    base = simulate(q, base_mode)
    var = simulate(qq, idea.get("mode", base_mode), idea["cfg"], score_col=score)
    dev_b, dev_v = stats(base.loc[DEV_START:DEV_END]), stats(var.loc[DEV_START:DEV_END])
    entry = {"id": idea["id"], "datum": datetime.now().strftime("%Y-%m-%d"), "titel": idea["titel"],
             "fraga": idea["fraga"], "portfolj": PORTS.get(base_mode, base_mode),
             "utveckling": {"fore": dev_b, "efter": dev_v}}
    if dev_v["sharpe"] - dev_b["sharpe"] >= MIN_GAIN:
        lk_b, lk_v = stats(base.loc[LOCKED_START:]), stats(var.loc[LOCKED_START:])
        entry["kontroll"] = {"fore": lk_b, "efter": lk_v}
        ok = lk_v["sharpe"] - lk_b["sharpe"] >= MIN_GAIN
        entry["beslut"] = "klarade" if ok else "foll_i_kontroll"
    else:
        entry["beslut"] = "forkastad"
    entry["sammanfattning"] = summarize_entry(entry)
    lab.append(entry)
    json.dump(lab, open(F_LAB, "w"), ensure_ascii=False, indent=1)
    print(f"   → {entry['beslut']}: {entry['sammanfattning']}")


def summarize_entry(e) -> str:
    f, a = e["utveckling"]["fore"], e["utveckling"]["efter"]
    fp = lambda x: f"{x * 100:+.1f}".replace(".", ",").replace("-", "−") + " %"
    txt = (f"På {DEV_START[:4]}–2023 gav idén {fp(a['arlig'])} per år mot {fp(f['arlig'])} för nuvarande "
           f"{e['portfolj']}, och värsta fallet blev {fp(a['max_ras'])} mot {fp(f['max_ras'])}.")
    if e["beslut"] == "forkastad":
        txt += " Det räckte inte som förbättring, så idén förkastas."
    elif e["beslut"] == "foll_i_kontroll":
        k = e["kontroll"]
        txt += (f" Den såg lovande ut, men på kontrollperioden 2024–idag räckte den inte "
                f"({fp(k['efter']['arlig'])} mot {fp(k['fore']['arlig'])} per år). "
                "Troligen var förbättringen tur.")
    else:
        k = e["kontroll"]
        txt += (f" Den höll även på kontrollperioden 2024–idag ({fp(k['efter']['arlig'])} mot "
                f"{fp(k['fore']['arlig'])} per år). Nästa steg är att prova den med låtsaspengar.")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            import anthropic
            msg = anthropic.Anthropic(api_key=key).messages.create(
                model=CLAUDE_MODEL, max_tokens=300, messages=[{"role": "user", "content":
                    "Förklara för en nybörjare, på 2–3 korta meningar på svenska, vad det här testet av en "
                    "tradingbot visade och vad man kan lära sig av det. Inga rubriker, ingen hype.\n\n"
                    f"Fråga: {e['fraga']}\nResultat: {txt}"}])
            return msg.content[0].text.strip()
        except Exception:
            pass
    return txt


# ---------------- Nyheter ----------------
def fetch_news(ticker: str) -> list[dict]:
    import yfinance as yf
    cutoff = datetime.now(timezone.utc) - timedelta(days=NEWS_DAYS)
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    items = []
    for n in raw:
        c = n.get("content", n)
        try:
            dt = datetime.fromisoformat((c.get("pubDate") or "").replace("Z", "+00:00"))
        except Exception:
            continue
        if dt >= cutoff and c.get("title"):
            items.append({"rubrik": c["title"], "sammanfattning": c.get("summary") or "",
                          "datum": dt.strftime("%Y-%m-%d")})
    return items


def sentiment_claude(ticker, items):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or not items:
        return None
    try:
        import anthropic
        listing = "\n".join(f"{i}. {x['rubrik']} — {x['sammanfattning']}" for i, x in enumerate(items))
        msg = anthropic.Anthropic(api_key=key).messages.create(
            model=CLAUDE_MODEL, max_tokens=1500, messages=[{"role": "user", "content":
                f"Du bedömer nyheter för aktien {ticker}. För varje nyhet: avgör om den verkligen "
                f"handlar om just det bolaget (t.ex. är Volvo Cars INTE AB Volvo VOLV-B.ST), och ge "
                f"en poäng -1 till 1 för hur den påverkar aktiekursen, plus kort motivering på svenska."
                f"\n\n{listing}\n\nSvara ENDAST med JSON: "
                f'[{{"nr":0,"relevant":true,"poang":0.3,"motivering":"..."}}]'}])
        text = msg.content[0].text
        for r in json.loads(text[text.index("["):text.rindex("]") + 1]):
            i = int(r["nr"])
            if 0 <= i < len(items):
                rel = bool(r.get("relevant", True))
                items[i].update(poang=float(r["poang"]) if rel else 0.0, relevant=rel,
                                motivering=r.get("motivering", ""), metod="Claude")
        return items
    except Exception as e:
        print(f"   Claude-analys misslyckades ({e}), använder VADER")
        return None


def sentiment_vader(items):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    for x in items:
        x.update(poang=sia.polarity_scores(f"{x['rubrik']}. {x['sammanfattning']}")["compound"],
                 relevant=True, motivering="", metod="VADER")
    return items


def news_score(ticker):
    items = fetch_news(ticker)
    if not items:
        return 0.0, []
    items = sentiment_claude(ticker, items) or sentiment_vader(items)
    rel = [x["poang"] for x in items if x.get("relevant", True)]
    return (float(np.mean(rel)) if rel else 0.0), items


# ---------------- Papperstrading ----------------
def new_portfolio():
    return {"version": VERSION, "start": datetime.now().strftime("%Y-%m-%d"),
            "portfoljer": {k: {"kassa": START_CASH, "innehav": {}, "senast_ombalans": None,
                               "senaste_filter": None} for k in PORTS},
            "signaler": [], "marknad": {}}


def load_portfolio():
    if os.path.exists(F_PORTFOLIO):
        pf = json.load(open(F_PORTFOLIO))
        if pf.get("version") == VERSION:
            return pf
        if pf.get("version") in (2, 3, 4, 5, 6):
            # Behåll befintliga portföljer, lägg till de som saknas
            for k in PORTS:
                pf["portfoljer"].setdefault(k, {"kassa": START_CASH, "innehav": {},
                                                "senast_ombalans": None, "senaste_filter": None})
            pf["version"] = VERSION
            print("Uppgraderar portföljfilen.")
            return pf
        print("Gammal portfölj hittad – startar om med nya portföljer.")
        for f in (F_TRADES, F_HISTORY, F_PRED):
            if os.path.exists(f):
                os.replace(f, f.replace(".csv", "_v1.csv"))
    return new_portfolio()


def evaluate_predictions(panel):
    if not os.path.exists(F_PRED):
        return
    pred = pd.read_csv(F_PRED)
    if "utfall" not in pred:
        return
    closes = panel.pivot(index="Date", columns="aktie", values="pris")
    for i, r in pred[pred["utfall"].isna()].iterrows():
        if r["aktie"] not in closes:
            continue
        after = closes[r["aktie"]][closes.index > pd.Timestamp(r["datum"])].dropna()
        if len(after) >= HORIZON:
            pred.loc[i, "utfall"] = float(after.iloc[HORIZON - 1] / r["pris"] - 1)
    done = pred.dropna(subset=["utfall"]).copy()
    if len(done):
        med = done.groupby("datum")["utfall"].transform("median")
        beat = done["utfall"] > med
        pred.loc[done.index, "ratt_ml"] = ((done["ml"] >= done.groupby("datum")["ml"]
                                            .transform("median")) == beat).astype(int)
        pred.loc[done.index, "ratt_med_nyheter"] = ((done["poang"] >= done.groupby("datum")["poang"]
                                                     .transform("median")) == beat).astype(int)
    pred.to_csv(F_PRED, index=False)


def append_csv(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def cmd_papper():
    pf = load_portfolio()
    fx = float(load_data("USDSEK=X")["Close"].iloc[-1])
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Papperstrading {today} (USD/SEK {fx:.2f})")
    panel, feats = build_panel()
    last_date = panel["Date"].max().strftime("%Y-%m-%d")
    if pf.get("senaste_data") == last_date:
        print(f"   Ingen ny handelsdag sedan {last_date} – inga affärer idag")
        return
    panel = add_short_interest(panel)
    evaluate_predictions(panel)

    # Träna AI-modellen på all historik med kända utfall
    train = panel.dropna(subset=["y"])
    model = new_model().fit(train[feats], train["y"])
    last = panel.sort_values("Date").groupby("aktie").tail(1).set_index("aktie")
    for c in RANK_COLS:  # senaste raden kan vara från olika dagar – räkna om rangordningen
        last[c + "_rank"] = last.groupby("se")[c].rank(pct=True)
    last["ml"] = model.predict_proba(last[feats])[:, 1]
    sample = train[train["Date"] > train["Date"].max() - pd.Timedelta(days=365)]
    sample = sample.sample(min(4000, len(sample)), random_state=0)
    imp = permutation_importance(model, sample[feats], sample["y"], n_repeats=3,
                                 random_state=0, scoring="roc_auc")
    top_feats = sorted(zip(feats, imp.importances_mean), key=lambda z: -z[1])[:8]
    # Riskmodellen: hur mycket väntas varje aktie svänga kommande månad?
    rf = risk_feats(feats)
    trv = panel.dropna(subset=["log_fwd_vol"])
    pred_vol = pd.Series(np.exp(new_risk_model().fit(trv[rf], trv["log_fwd_vol"]).predict(last[rf])),
                         index=last.index)

    print("   läser nyheter...")
    sigs, archive = [], []
    for t, r in last.iterrows():
        sent, items = news_score(t)
        archive += [{"hamtad": today, "aktie": t, "publicerad": x["datum"], "rubrik": x["rubrik"],
                     "sammanfattning": x["sammanfattning"], "poang": round(x["poang"], 3),
                     "relevant": x.get("relevant", True), "metod": x.get("metod", "")} for x in items]
        sigs.append({"aktie": t, "datum": r["Date"].strftime("%Y-%m-%d"), "pris": float(r["pris"]),
                     "ml": float(r["ml"]), "nyheter": sent,
                     "poang": float(np.clip(r["ml"] + NEWS_WEIGHT * sent, 0, 1)),
                     "risk": float(pred_vol.get(t, np.nan)),
                     "blankning": None if pd.isna(r.get("blankning", np.nan)) else float(r["blankning"]),
                     "nyhetslista": items[:4]})
    # Nyhetsarkiv – samlas för framtida träning
    if archive:
        new = pd.DataFrame(archive)
        if os.path.exists(F_NEWS):
            old = pd.read_csv(F_NEWS)
            new = new[~(new["aktie"] + new["rubrik"]).isin(old["aktie"] + old["rubrik"])]
        append_csv(F_NEWS, new.to_dict("records"))
    # Affärerna styrs bara av den testade AI-modellen. Nyheterna sparas och utvärderas separat
    # (facit "med nyheter") innan de får påverka något.
    scores = pd.Series({s["aktie"]: s["ml"] for s in sigs})
    blank = last["blankning"] if "blankning" in last else None
    prices = {s["aktie"]: to_sek(s["aktie"], s["pris"], fx) for s in sigs}
    # Om en kurs saknas idag: använd senast kända pris, och ombalansera inte på ofullständig data
    known = {**pf.get("senaste_priser", {}), **prices}
    missing = [t for p_ in pf["portfoljer"].values() for t in p_.get("innehav", {}) if t not in prices]
    if missing:
        print(f"   Kurs saknas idag för {sorted(set(missing))} – använder senast kända pris, ingen ombalansering")
    prices = known

    mk = {}
    for key, tick in (("us", MARKET_US), ("se", MARKET_SE)):
        c = load_data(tick)["Close"]
        mk[key] = float(c.iloc[-1] / c.rolling(200).mean().iloc[-1] - 1)
    below = {k: v < 0 for k, v in mk.items()}

    all_trades, hist_row = [], {"datum": today}
    for key, label in PORTS.items():
        port = pf["portfoljer"][key]
        hold = {t: float(n) for t, n in port["innehav"].items()}
        if key == "hold":
            # Gör ingenting: köper alla aktier lika mycket på startdagen och rör dem sedan aldrig
            if not hold:
                d0 = pf["start"]
                fx0 = close_on("USDSEK=X", d0)
                each = START_CASH / len(scores)
                for t in scores.index:
                    p0 = to_sek(t, close_on(t, d0), fx0)
                    hold[t] = each * (1 - cost_of(t)) / p0
                    all_trades.append({"datum": d0, "portfolj": label, "aktie": t, "handling": "KÖP",
                                       "belopp_sek": round(each, 2), "vikt": round(1 / len(scores), 4)})
                port["kassa"] = 0.0
                port["senast_ombalans"] = d0
                port["vikter"] = {t: round(1 / len(scores), 4) for t in scores.index}
            port["innehav"] = hold
            equity = sum(n * prices.get(t, 0) for t, n in hold.items())
            hist_row[key] = round(equity, 2)
            print(f"   {label}: {equity:,.0f} kr (rör aldrig något)")
            continue
        equity = port["kassa"] + sum(n * prices.get(t, 0) for t, n in hold.items())
        last_reb = port["senast_ombalans"]
        due = (last_reb is None or (pd.Timestamp(today) - pd.Timestamp(last_reb)).days >= 28
               or port["senaste_filter"] != below) and not missing
        if due:
            w = target_weights(scores, below, key, pred_vol, blank)
            for t in sorted(w.index, key=lambda t: w[t] * equity - hold.get(t, 0) * prices[t]):
                diff = w[t] * equity - hold.get(t, 0) * prices[t]
                if abs(diff) < MIN_TRADE * equity:
                    continue
                if diff > 0:
                    diff = min(diff, port["kassa"] / (1 + cost_of(t)))
                    if diff <= 0:
                        continue
                n = diff / prices[t]
                hold[t] = hold.get(t, 0) + n
                port["kassa"] -= diff + abs(diff) * cost_of(t)
                all_trades.append({"datum": today, "portfolj": label, "aktie": t,
                                   "handling": "KÖP" if diff > 0 else "SÄLJ",
                                   "belopp_sek": round(abs(diff), 2), "vikt": round(float(w[t]), 4)})
            hold = {t: n for t, n in hold.items() if n * prices.get(t, 1e9) > 1}
            port["senast_ombalans"], port["senaste_filter"] = today, below
            port["vikter"] = {t: round(float(v), 4) for t, v in w.items()}
        port["innehav"] = hold
        equity = port["kassa"] + sum(n * prices.get(t, 0) for t, n in hold.items())
        hist_row[key] = round(equity, 2)
        print(f"   {label}: {equity:,.0f} kr ({'ombalanserad' if due else 'oförändrad'})")

    spy = prices.get("SPY") or to_sek("SPY", float(load_data("SPY")["Close"].iloc[-1]), fx)
    pf.setdefault("spy_start", spy)
    hist_row["jamforelse_spy"] = round(START_CASH * spy / pf["spy_start"], 2)
    # Global indexfond som måttstock, räknad från samma startdag som portföljerna
    try:
        idx_now = close_on(INDEX_FUND, today) * fx
        pf.setdefault("index_start", close_on(INDEX_FUND, pf["start"]) * close_on("USDSEK=X", pf["start"]))
        hist_row["jamforelse_index"] = round(START_CASH * idx_now / pf["index_start"], 2)
    except Exception as e:
        print(f"   Indexfonden gick inte att hämta ({e})")

    pf["senaste_priser"] = {t: float(v) for t, v in prices.items()}
    pf["senaste_data"] = last_date
    pf.update(signaler=sigs, marknad=mk, viktigast=[(n, float(v)) for n, v in top_feats],
              uppdaterad=datetime.now().isoformat(timespec="minutes"))
    json.dump(pf, open(F_PORTFOLIO, "w"), ensure_ascii=False, indent=1)
    append_csv(F_TRADES, all_trades)
    hist = pd.read_csv(F_HISTORY) if os.path.exists(F_HISTORY) else pd.DataFrame()
    if len(hist) and "datum" in hist:
        hist = hist[hist["datum"] != today]
    pd.concat([hist, pd.DataFrame([hist_row])]).to_csv(F_HISTORY, index=False)
    old = pd.read_csv(F_PRED) if os.path.exists(F_PRED) else pd.DataFrame()
    new = pd.DataFrame([{k: s[k] for k in ("datum", "aktie", "pris", "ml", "nyheter", "poang")}
                        for s in sigs])
    if len(old):
        new = new[~new.set_index(["datum", "aktie"]).index.isin(old.set_index(["datum", "aktie"]).index)]
    pd.concat([old, new]).to_csv(F_PRED, index=False)
    print(f"   {len(all_trades)} affärer idag")


# ---------------- Rapport (webbappen) ----------------
def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


PORT_INFO = {
    "ai": {"namn": "AI-portföljen", "farg": "#3b5bdb",
           "kort": "Lägger mer pengar i aktierna AI:n tror mest på."},
    "enkel": {"namn": "Enkla portföljen", "farg": "#0c8599",
              "kort": "Lika mycket i alla 33 aktier. Ingen AI."},
    "risk": {"namn": "Risk-portföljen", "farg": "#e67700",
             "kort": "Mindre pengar i aktier som AI:n tror kommer svänga mycket."},
    "hold": {"namn": "Gör ingenting", "farg": "#6c757d",
             "kort": "Köpte alla 33 aktier första dagen och rör dem aldrig. Måttstocken att slå."},
}

FEATURE_NAMES = {
    "avk_1d": "gårdagens rörelse", "avk_5d": "senaste veckan", "avk_10d": "senaste två veckorna",
    "avk_20d": "senaste månaden", "avk_60d": "senaste kvartalet", "avk_120d": "senaste halvåret",
    "avst_ma10": "läge mot 10-dagars snitt", "avst_ma20": "läge mot 20-dagars snitt",
    "avst_ma50": "läge mot 50-dagars snitt", "avst_ma200": "läge mot 200-dagars snitt",
    "vol_20": "svängningar senaste månaden", "vol_60": "svängningar senaste kvartalet",
    "vol_kvot": "om svängningarna ökar", "rsi_14": "RSI (överköpt/översåld)",
    "macd_hist": "MACD (trendskifte)", "atr_pct": "dagliga kursspann", "volym_z": "ovanlig handelsvolym",
    "fran_arshogsta": "avstånd till årshögsta", "marknad_avk_20d": "börsens senaste månad",
    "marknad_avst_ma200": "börsens läge mot 200-dagars snitt",
    "relativ_styrka_60d": "styrka jämfört med börsen", "se": "svensk eller amerikansk aktie",
}


def feature_name(n: str) -> str:
    base = n.replace("_rank", "")
    name = FEATURE_NAMES.get(base, base)
    return name + (" (jämfört med de andra)" if n.endswith("_rank") else "")


def kr(x) -> str:
    return f"{x:,.0f}".replace(",", "\u00a0") + "\u00a0kr"


def signed_pct(x, d=1) -> str:
    return f"{x:+.{d}f} %".replace(".", ",").replace("-", "−")


def luck_test(h: pd.DataFrame, port: str, base: str, min_days: int = 20) -> dict:
    """Skicklighet eller tur? Jämför en portfölj mot en bas (t.ex. enkla portföljen) dag för dag.
    Slumptest: vi slår om tecknet på varje dags över-/underprestation slumpmässigt 5000 gånger.
    p = hur ofta ren tur ger minst lika bra resultat som det boten faktiskt fick."""
    if port not in h or base not in h:
        return {"dagar": 0, "klar": False}
    v = h[[port, base]].dropna()
    r = v.pct_change().dropna()
    ex = (r[port] - r[base]).to_numpy()
    n = len(ex)
    out = {"dagar": n, "klar": n >= min_days, "snitt": float(ex.mean()) if n else 0.0}
    if n < min_days or ex.std(ddof=1) == 0:
        out["behover"] = min_days
        return out
    rng = np.random.default_rng(42)
    flips = rng.choice([-1.0, 1.0], size=(5000, n))
    perm = (flips * ex).mean(axis=1)
    out["p"] = float((perm >= ex.mean()).mean())
    out["t"] = float(ex.mean() / (ex.std(ddof=1) / np.sqrt(n)))
    if ex.mean() > 0:
        out["dagar_till_saker"] = int(np.ceil((1.96 * ex.std(ddof=1) / ex.mean()) ** 2))
    return out


def luck_section(h: pd.DataFrame) -> str:
    """HTML-sektion 'Skicklighet eller tur?' för rapporten."""
    rows = ""
    for port, base, label in (("ai", "enkel", "AI-portföljen mot enkla portföljen"),
                              ("risk", "enkel", "Risk-portföljen mot enkla portföljen")):
        t = luck_test(h, port, base)
        if not t["dagar"]:
            continue
        if not t["klar"]:
            verdict = (f"För tidigt att säga. {t['dagar']} av minst {t['behover']} handelsdagar "
                       "har gått. Skillnaden hittills är mest slump.")
        else:
            p = t["p"]
            if p < 0.05 and t["snitt"] > 0:
                verdict = f"Troligen skicklighet. Chansen att det är ren tur är ca {p * 100:.0f} %."
            elif t["snitt"] <= 0:
                verdict = f"Ingen fördel hittills. Portföljen ligger inte över basen (tur-chans {p * 100:.0f} %)."
            else:
                verdict = f"Kan lika gärna vara tur. Chansen att det är ren tur är ca {p * 100:.0f} %."
            if "dagar_till_saker" in t:
                verdict += f" Med dagens tempo behövs ca {t['dagar_till_saker']} handelsdagar för att vara säker."
        rows += f"<li><b>{esc(label)}</b><br><span class='desc'>{esc(verdict)}</span></li>"
    if not rows:
        return ""
    return ("<section aria-labelledby='luck-h'><h2 id='luck-h'>Skicklighet eller tur?</h2>"
            "<p class='lede'>Att ligga före en dag säger inget. Här testar vi om försprånget är större än vad "
            "slumpen ensam brukar ge. Först efter minst 20 handelsdagar ger testet ett svar, och ju längre desto säkrare.</p>"
            f"<ul class='luck'>{rows}</ul></section>")


def cmd_rapport():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pf = json.load(open(F_PORTFOLIO)) if os.path.exists(F_PORTFOLIO) else None
    h = pd.read_csv(F_HISTORY) if os.path.exists(F_HISTORY) else pd.DataFrame()
    bt = json.load(open(F_BACKTEST)) if os.path.exists(F_BACKTEST) else {}
    if bt.get("version") != VERSION:
        bt = {}
    keys = [k for k in PORTS if len(h) and k in h]
    sec = []          # HTML-sektioner
    chart_data = {}   # data till graferna

    # ---- Loppet ----
    if keys:
        last = h.iloc[-1]
        vals = {k: float(last[k]) for k in keys if pd.notna(last[k])}
        ranking = sorted(vals, key=lambda k: -vals[k])
        spread = (vals[ranking[0]] - vals[ranking[-1]]) / START_CASH
        leader = PORT_INFO[ranking[0]]["namn"]
        days = len(h)
        if spread < 0.002:
            headline = "Jämnt lopp än så länge"
            sub = (f"Efter {days} {'dag' if days == 1 else 'dagar'} skiljer det mindre än 0,2 % mellan "
                   "portföljerna. Det behövs några veckor innan det betyder något.")
        else:
            headline = f"{leader} leder"
            sub = (f"Efter {days} handelsdagar ligger den {signed_pct(spread * 100)} före sista plats. "
                   "Kortare perioder än ett par månader säger ofta mest om tur.")
        col = "jamforelse_index" if "jamforelse_index" in last and pd.notna(last["jamforelse_index"]) else "jamforelse_spy"
        spy = float(last[col]) if col in last and pd.notna(last[col]) else None
        lo = min(list(vals.values()) + ([spy] if spy else [])) / START_CASH - 1
        hi = max(list(vals.values()) + ([spy] if spy else [])) / START_CASH - 1
        span = max(abs(lo), abs(hi), 0.01)
        lanes = ""
        for pos, k in enumerate(ranking, 1):
            info, v = PORT_INFO[k], vals[k]
            r = v / START_CASH - 1
            width = 50 + 50 * r / span  # 50 % = startlinjen
            lanes += f"""
<li class="lane" style="--c:{info['farg']}">
  <div class="lane-head"><span class="place">{pos}</span><span class="pname">{info['namn']}</span>
    <span class="pval">{kr(v)}</span></div>
  <div class="track"><div class="startline"></div><div class="runner" style="left:{width:.1f}%"></div></div>
  <div class="lane-foot"><span class="{'up' if r >= 0 else 'down'}">{signed_pct(r * 100, 2)}</span>
    <span class="desc">{info['kort']}</span></div>
</li>"""
        bench = ("en global indexfond (hela världens börser)" if col == "jamforelse_index"
                 else "hela den amerikanska börsen (SPY)")
        spy_txt = (f"<p class='pace'>Som jämförelse: 100 000 kr i {bench} "
                   f"hade varit värt <b>{kr(spy)}</b> ({signed_pct((spy / START_CASH - 1) * 100, 2)}).</p>"
                   if spy else "")
        sec.append(f"""
<section class="race" aria-labelledby="race-h">
  <p class="kicker">Fyra portföljer, 100&nbsp;000 låtsaskronor var, start {esc(pf['start'] if pf else '')}</p>
  <h1 id="race-h">{headline}</h1>
  <p class="lede">{sub}</p>
  <ol class="lanes">{lanes}</ol>
  {spy_txt}
  <figure class="chart-wrap {'hidden' if days < 2 else ''}"><canvas id="raceChart" height="220"
    aria-label="Portföljernas värde dag för dag"></canvas></figure>
</section>""")
        chart_data["race"] = {"labels": h["datum"].tolist(),
                              "series": [{"namn": PORT_INFO[k]["namn"], "farg": PORT_INFO[k]["farg"],
                                          "data": [None if pd.isna(x) else round(float(x)) for x in h[k]]}
                                         for k in keys],
                              "spy": [None if pd.isna(x) else round(float(x)) for x in h.get(col, [])],
                              "spy_namn": "Global indexfond" if col == "jamforelse_index" else "SPY"}

    # ---- Skicklighet eller tur? ----
    if len(h):
        sec.append(luck_section(h))

    # ---- Idag i korthet ----
    if pf:
        mk = pf.get("marknad", {})
        lights = ""
        for k, n in (("us", "USA-börsen"), ("se", "Stockholmsbörsen")):
            if k in mk:
                ok = mk[k] >= 0
                lights += (f"<li class='light {'go' if ok else 'stop'}'><span class='dot' aria-hidden='true'></span>"
                           f"<div><b>{n}: {'grönt ljus' if ok else 'rött ljus'}</b><br>"
                           f"{'Ligger' if ok else 'Ligger'} {signed_pct(abs(mk[k]) * 100)} "
                           f"{'över' if ok else 'under'} sitt 200-dagars snitt, så alla portföljer har "
                           f"{'full insats' if ok else 'bara halv insats'} där.</div></li>")
        trades_today = 0
        if os.path.exists(F_TRADES):
            tr = pd.read_csv(F_TRADES)
            if "datum" in tr:
                trades_today = int((tr["datum"] == tr["datum"].max()).sum())
        sig = sorted(pf.get("signaler", []), key=lambda s: -s["ml"])
        nxt = []
        for k in keys:
            lr = pf["portfoljer"][k].get("senast_ombalans")
            if lr:
                nxt.append(pd.Timestamp(lr) + pd.Timedelta(days=28))
        nxt_txt = f"Nästa planerade ombalansering: runt {min(nxt):%-d %B}.".replace(
            "January", "januari").replace("February", "februari").replace("March", "mars").replace(
            "April", "april").replace("May", "maj").replace("June", "juni").replace("July", "juli").replace(
            "August", "augusti").replace("September", "september").replace("October", "oktober").replace(
            "November", "november").replace("December", "december") if nxt else ""
        sec.append(f"""
<section aria-labelledby="today-h">
  <h2 id="today-h">Läget idag</h2>
  <ul class="lights">{lights}</ul>
  <p>Senaste handelsdagen gjordes <b>{trades_today} affärer</b>. Portföljerna byter bara innehav en gång i
  månaden, eller direkt om ett ljus slår om. {nxt_txt}</p>
</section>""")

        # ---- AI:ns favoriter ----
        if sig:
            vik = pf["portfoljer"]["ai"].get("vikter", {})

            def row(s):
                p = s["ml"]
                news = s.get("nyheter", 0)
                wtxt = f"{vik.get(s['aktie'], 0) * 100:.1f}".replace(".", ",")
                ntag = ("<span class='tag pos'>bra nyheter</span>" if news > 0.3 else
                        "<span class='tag neg'>dåliga nyheter</span>" if news < -0.3 else "")
                if (s.get("blankning") or 0) >= BLANK_LIMIT:
                    ntag += (f"<span class='tag neg'>blankad "
                             f"{str(round(s['blankning'], 1)).replace('.', ',')} %</span>")
                return (f"<li><span class='tick'>{esc(s['aktie'].replace('.ST', ''))}</span>"
                        f"<span class='meter' style='--p:{p * 100:.0f}%'><span></span></span>"
                        f"<span class='num'>{p * 100:.0f} %</span>{ntag}"
                        f"<span class='w'>{wtxt} % av AI-portföljen</span></li>")
            top = "".join(row(s) for s in sig[:5])
            bottom = "".join(row(s) for s in sig[-5:][::-1])
            full = "".join(row(s) for s in sig)
            seen = []
            for n, _ in pf.get("viktigast", []):
                fn = FEATURE_NAMES.get(n.replace("_rank", ""), n)
                if fn not in seen:
                    seen.append(fn)
            feats = seen[:4]
            feat_txt = ("<p class='note'>Det AI:n tittar mest på just nu: " +
                        ", ".join(esc(f) for f in feats) + ".</p>") if feats else ""
            sec.append(f"""
<section aria-labelledby="fav-h">
  <h2 id="fav-h">AI:ns favoriter</h2>
  <p>Procenten är AI:ns bedömning av chansen att aktien går bättre än mittenaktien den kommande månaden.
  50 % betyder att den inte vet. Över 55 % är ovanligt säkert för den här modellen.</p>
  <h3>Tror mest på</h3><ul class="bars">{top}</ul>
  <h3>Tror minst på</h3><ul class="bars">{bottom}</ul>
  <details><summary>Visa alla {len(sig)} aktier</summary><ul class="bars">{full}</ul></details>
  {feat_txt}
</section>""")

            # ---- Nyheter ----
            items = [(s["aktie"], n) for s in sig for n in s.get("nyhetslista", [])
                     if n.get("relevant", True)]
            items.sort(key=lambda x: -abs(x[1]["poang"]))
            if items:
                def news_li(a, n):
                    cls = "pos" if n["poang"] > 0.1 else "neg" if n["poang"] < -0.1 else "neu"
                    label = "positiv" if cls == "pos" else "negativ" if cls == "neg" else "neutral"
                    why = f"<span class='why'>{esc(n['motivering'])}</span>" if n.get("motivering") else ""
                    return (f"<li class='{cls}'><span class='sent'>{label}</span>"
                            f"<span class='nt'><b>{esc(a.replace('.ST', ''))}</b> {esc(n['rubrik'])}</span>{why}</li>")
                first = "".join(news_li(a, n) for a, n in items[:6])
                rest = "".join(news_li(a, n) for a, n in items[6:30])
                metod = items[0][1].get("metod", "")
                how = ("Claude läser nyheterna och bedömer dem." if metod == "Claude" else
                       "Nyheterna bedöms med en enkel ordlista (VADER), som ibland missförstår rubriker.")
                sec.append(f"""
<section aria-labelledby="news-h">
  <h2 id="news-h">Nyheter boten har läst</h2>
  <p>De starkaste nyheterna från senaste veckan. Nyheterna påverkar inte affärerna än:
  boten sparar dem och mäter först om de hade hjälpt.
  {how}</p>
  <ul class="news">{first}</ul>
  {f'<details><summary>Fler nyheter</summary><ul class="news">{rest}</ul></details>' if rest else ''}
</section>""")

    # ---- Facit ----
    facit = ""
    if os.path.exists(F_PRED):
        p = pd.read_csv(F_PRED)
        if "ratt_ml" in p and p["ratt_ml"].notna().any():
            d = p.dropna(subset=["ratt_ml"])
            hit = d["ratt_ml"].mean()
            verdict = ("bättre än slumpen" if hit > 0.53 else "sämre än slumpen" if hit < 0.47
                       else "ungefär som slumpen")
            facit = (f"<p class='big'>{hit * 100:.0f} %</p><p>av AI:ns {len(d)} gissningar har slagit in, "
                     f"vilket är <b>{verdict}</b> (50 %). Med nyheterna inräknade: "
                     f"{d['ratt_med_nyheter'].mean() * 100:.0f} %.</p>")
    if not facit:
        facit = (f"<p>Varje dag gissar AI:n vilka aktier som går bäst den kommande månaden. Efter "
                 f"{HORIZON} handelsdagar, ungefär en månad, kan vi se hur ofta den hade rätt. "
                 f"Första facit kommer i mitten av december.</p>")
    sec.append(f"<section aria-labelledby='facit-h'><h2 id='facit-h'>Facit: har AI:n rätt?</h2>{facit}</section>")

    # ---- AI-forskarens labbdagbok ----
    lab = load_lab()
    done = {e["id"] for e in lab}
    nxt = next((i for i in IDEAS if i["id"] not in done), None)
    status = {"forkastad": ("Förkastad", "neg"), "foll_i_kontroll": ("Föll i kontrollen", "neg"),
              "klarade": ("Lovande – prova live", "pos")}
    entries = ""
    for e in reversed(lab):
        lab_txt, cls = status.get(e["beslut"], (e["beslut"], ""))
        entries += (f"<li><div class='lab-head'><b>{esc(e['titel'])}</b><span class='tag {cls}'>{lab_txt}</span></div>"
                    f"<p class='lab-q'>{esc(e['fraga'])}</p><p class='lab-a'>{esc(e['sammanfattning'])}</p>"
                    f"<small>{e['datum']}</small></li>")
    if not entries:
        entries = "<li><p>Första experimentet körs nästa måndag kväll.</p></li>"
    nxt_txt = (f"<p class='note'>Nästa experiment: <b>{esc(nxt['titel'])}</b>. {esc(nxt['fraga'])}</p>"
               if nxt else "<p class='note'>Alla idéer på listan är testade.</p>")
    sec.append(f"""
<section aria-labelledby="lab-h">
  <h2 id="lab-h">AI-forskarens labbdagbok</h2>
  <p>Varje måndag testar boten en ny idé för att bli bättre. Reglerna är stränga så att den inte lurar sig själv:
  idéerna är bestämda i förväg, de prövas först på 2003–2023, och bara de som klarar det kontrolleras på
  2024–idag. Klarar en idé båda är nästa steg att prova den med låtsaspengar, eftersom det är det enda
  provet som är helt fritt från efterhandskunskap. Inget ändras i portföljerna utan att du och jag bestämmer det.</p>
  <ul class="lab">{entries}</ul>
  {nxt_txt}
</section>""")

    # ---- Historiktest ----
    if bt:
        names = [("AI-portföljen", PORT_INFO["ai"]["farg"]), ("Enkla portföljen", PORT_INFO["enkel"]["farg"]),
                 ("Risk-portföljen", PORT_INFO["risk"]["farg"]), ("Gör ingenting", PORT_INFO["hold"]["farg"]),
                 ("USA-börsen (SPY)", "#adb5bd")]
        chart_data["bt"] = {"labels": bt["kurvor"]["datum"],
                            "series": [{"namn": n, "farg": c, "data": [round(x * 100_000) for x in bt["kurvor"][n]]}
                                       for n, c in names if n in bt["kurvor"]]}
        rows = ""
        for n, c in names:
            s = bt["strategier"].get(n)
            if s:
                rows += (f"<tr><th scope='row'><span class='sw' style='background:{c}'></span>{n}</th>"
                         f"<td>{signed_pct(s['hela']['arlig'] * 100)}</td>"
                         f"<td>{signed_pct(s['hela']['max_ras'] * 100, 0)}</td></tr>")
        L = pd.DataFrame(bt["larande"])
        chart_data["auc"] = {"labels": L["period"].tolist(), "data": [round(a, 3) for a in L["auc"]]}
        if "risk_fel_ai" in L:
            chart_data["risk"] = {"labels": L["period"].tolist(),
                                  "ai": [round(a * 100, 1) for a in L["risk_fel_ai"]],
                                  "enkel": [round(a * 100, 1) for a in L["risk_fel_enkel"]]}
        better = int((L["auc"] > 0.5).sum())
        # Kriserna
        crisis_html = ""
        if bt.get("kriser"):
            head = "".join(f"<th scope='col'>{n.replace('-portföljen', '').replace(' portföljen', '')}</th>"
                           for n, _ in names)
            body = ""
            for crisis, row in bt["kriser"].items():
                cells = ""
                for n, _ in names:
                    v = row.get(n)
                    cells += ("<td class='na'>–</td>" if v is None else
                              f"<td class='{'up' if v >= 0 else 'down'}'>{signed_pct(v * 100, 0)}</td>")
                body += f"<tr><th scope='row'>{crisis}</th>{cells}</tr>"
            crisis_html = f"""
  <h3>Hur gick det i kriserna?</h3>
  <p>Hur mycket varje portfölj tappade från toppen till botten av de stora börsfallen. Det är här
  marknadsfiltret och riskmodellen ska göra nytta.</p>
  <div class="scroll"><table class="crisis"><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></div>
  <p class="note">Streck betyder att AI:n inte hade hunnit tränas än. Den behöver några års historik,
  så AI-portföljerna kan testas först från 2003.</p>"""
        # Säkerhetsmätaren
        sure_html = ""
        sk = bt.get("sakerhet", {})
        def years_txt(key):
            v = sk.get(key, {})
            y = v.get("ar")
            if y is None:
                return None, v
            return (f"{y:.0f} år" if y < 50 else "mer än 50 år"), v
        rows_s = ""
        for key, label in (("AI-portföljen|Gör ingenting", "AI-portföljen mot Gör ingenting"),
                           ("Enkla portföljen|Gör ingenting", "Enkla portföljen mot Gör ingenting"),
                           ("AI-portföljen|Enkla portföljen", "AI-portföljen mot Enkla portföljen")):
            yt, v = years_txt(key)
            if yt:
                rows_s += (f"<tr><th scope='row'>{label}</th><td>{signed_pct(v['skillnad'] * 100)}</td>"
                           f"<td><b>{yt}</b></td></tr>")
        if rows_s:
            sure_html = f"""
  <h3>Hur länge måste vi vänta?</h3>
  <p>Portföljerna går upp och ner av slumpen, så en skillnad måste vara stor och hålla länge innan vi kan
  lita på den. Tabellen visar hur länge vi skulle behöva följa portföljerna för att vara säkra, om skillnaden
  blir lika stor som i historiken.</p>
  <table class="bt"><thead><tr><th scope="col">Jämförelse</th><th scope="col">Skillnad per år</th>
  <th scope="col">Tid som behövs</th></tr></thead><tbody>{rows_s}</tbody></table>
  <p class="note">Därför ska vi inte dra slutsatser av några veckors eller månaders resultat.</p>"""
        sec.append(f"""
<section aria-labelledby="bt-h">
  <h2 id="bt-h">Så hade det gått sedan 2003</h2>
  <p>Ett test där varje portfölj bara fick se det som hade hänt fram till varje dag. Kurvan visar vad
  100&nbsp;000&nbsp;kr hade blivit. Courtage och valutaväxling är inräknat.</p>
  <figure class="chart-wrap"><canvas id="btChart" height="240" aria-label="Historiktest sedan 2003"></canvas></figure>
  <table class="bt"><thead><tr><th scope="col">Portfölj</th><th scope="col">Per år</th>
  <th scope="col">Värsta fall</th></tr></thead><tbody>{rows}</tbody></table>
  <p class="note">Värsta fall är hur mycket portföljen som mest tappade från en topp. Aktierna är dagens
  stora bolag, alltså de som lyckades. Det gör alla portföljernas siffror för bra: utan de fem största
  vinnarna sjunker avkastningen med flera procentenheter per år. USA-börsen (SPY) har inte det problemet
  och är därför en ärligare jämförelse.</p>
  {crisis_html}
  {sure_html}

  <h3>Kan AI:n välja vinnare?</h3>
  <p>Varje stapel är ett år som AI:n aldrig sett. Över mittlinjen betyder att den valde bättre än slumpen.
  Det lyckades {better} av {len(L)} år, och marginalen är liten.</p>
  <figure class="chart-wrap"><canvas id="aucChart" height="170" aria-label="AI:ns träffsäkerhet per år"></canvas></figure>
  {'''<h3>Kan AI:n förutse risk?</h3>
  <p>Hur fel riskmodellen gissade aktiernas svängningar, jämfört med att bara anta att de svänger som förra
  månaden. Lägre är bättre. Här är AI:n faktiskt lite bättre.</p>
  <figure class="chart-wrap"><canvas id="riskChart" height="170" aria-label="Riskmodellens fel per år"></canvas></figure>''' if "risk" in chart_data else ""}
</section>""")

    # ---- Affärer + ordlista ----
    if os.path.exists(F_TRADES):
        tr = pd.read_csv(F_TRADES)
        if "portfolj" in tr and len(tr):
            tr = tr.tail(40).iloc[::-1]
            rows = "".join(f"<tr><td>{r.datum}</td><td>{esc(r.portfolj.replace('-portföljen', '').replace(' portföljen', ''))}</td>"
                           f"<td class='{'buy' if r.handling == 'KÖP' else 'sell'}'>{r.handling.capitalize()}</td>"
                           f"<td>{esc(r.aktie.replace('.ST', ''))}</td><td class='r'>{kr(r.belopp_sek)}</td></tr>"
                           for r in tr.itertuples())
            sec.append(f"""<section aria-labelledby="tr-h"><h2 id="tr-h">Affärer</h2>
<details><summary>Visa de senaste {len(tr)} affärerna</summary>
<table class="trades"><tbody>{rows}</tbody></table></details></section>""")
    sec.append("""<section aria-labelledby="ord-h"><h2 id="ord-h">Ord som används</h2>
<dl class="glossary">
<dt>200-dagars snitt</dt><dd>Genomsnittspriset de senaste 200 handelsdagarna. Ligger börsen under det har
den ofta varit i en nedgång, och då tar portföljerna mindre risk.</dd>
<dt>Ombalansering</dt><dd>När portföljen säljer och köper för att komma tillbaka till sina önskade andelar.</dd>
<dt>Blankning</dt><dd>När någon lånar och säljer en aktie för att tjäna på att den faller. Finansinspektionen
publicerar alla stora blankningar. Aktier där minst 2 % av aktierna är blankade får bara 30 % av sin vanliga
vikt i alla portföljer, eftersom de i testerna har gått något sämre.</dd>
<dt>Courtage och valutaväxling</dt><dd>Avgifterna för varje affär. Boten räknar med 0,15 % för svenska
aktier och 0,40 % för amerikanska, eftersom banken också tar cirka 0,25 % för att växla kronor till dollar.</dd>
<dt>SPY</dt><dd>En fond som följer de 500 största bolagen i USA. Används som jämförelse.</dd>
<dt>Svängning (volatilitet)</dt><dd>Hur mycket en aktie brukar röra sig upp och ner.</dd>
</dl></section>""")

    updated = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = HTML_TEMPLATE.replace("{{UPDATED_ISO}}", datetime.now().strftime("%Y-%m-%dT%H:%M:%S")) \
        .replace("{{UPDATED}}", updated).replace("{{SECTIONS}}", "\n".join(sec)) \
        .replace("{{DATA}}", json.dumps(chart_data, ensure_ascii=False))
    open(F_REPORT, "w").write(html)
    docs = os.path.join(DIR, "docs")
    os.makedirs(docs, exist_ok=True)
    open(os.path.join(docs, "index.html"), "w").write(html)
    json.dump({"name": "Tradingbot", "short_name": "Tradingbot", "start_url": ".",
               "display": "standalone", "background_color": "#eef2f7", "theme_color": "#14213d",
               "icons": [{"src": "ikon.png", "sizes": "512x512", "type": "image/png"}]},
              open(os.path.join(docs, "manifest.json"), "w"))
    icon = os.path.join(docs, "ikon.png")
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    for y, c, x in ((0.28, PORT_INFO["risk"]["farg"], 0.62), (0.5, PORT_INFO["enkel"]["farg"], 0.78),
                    (0.72, PORT_INFO["ai"]["farg"], 0.7)):
        ax.plot([0.14, 0.86], [y, y], color="#33415c", lw=10, solid_capstyle="round")
        ax.scatter([x], [y], s=2600, color=c, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("#14213d")
    fig.savefig(icon); plt.close(fig)
    print("Rapport sparad: rapport.html + docs/")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="sv"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tradingbot</title>
<meta http-equiv="refresh" content="1800">
<link rel="manifest" href="manifest.json"><link rel="apple-touch-icon" href="ikon.png">
<meta name="apple-mobile-web-app-capable" content="yes"><meta name="theme-color" content="#14213d">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;600;700&family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&display=swap" rel="stylesheet">
<style>
:root{--bg:#eef2f7;--paper:#fff;--ink:#14213d;--soft:#4a5672;--line:#d6dde8;--up:#2b8a3e;--down:#c92a2a;
--ai:#3b5bdb;--enkel:#0c8599;--risk:#e67700;--track:#e3e9f2}
@media (prefers-color-scheme:dark){:root{--bg:#0f172a;--paper:#16213a;--ink:#e8edf6;--soft:#a7b3c9;--line:#2a3756;
--up:#51cf66;--down:#ff8787;--track:#24314f}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.55 "Figtree",system-ui,sans-serif;
-webkit-font-smoothing:antialiased}
.top{background:var(--ink);color:var(--bg);padding:14px 20px}
@media (prefers-color-scheme:dark){.top{background:#08101f;color:var(--ink)}}
.top-in{max-width:680px;margin:auto;display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
.brand{font:800 20px/1 "Bricolage Grotesque",sans-serif;letter-spacing:-.01em}
.top small{opacity:.75;font-size:14px}
main{max-width:680px;margin:auto;padding:8px 20px 60px}
section{padding:28px 0;border-bottom:1px solid var(--line)}
section:last-child{border-bottom:0}
h1,h2,h3{font-family:"Bricolage Grotesque",sans-serif;line-height:1.1;margin:0 0 10px}
h1{font-size:clamp(34px,9vw,48px);font-weight:800;letter-spacing:-.02em}
h2{font-size:24px;font-weight:800}
h3{font-size:17px;font-weight:600;margin-top:22px;color:var(--soft)}
p{margin:0 0 12px;max-width:62ch}
.kicker{color:var(--soft);font-size:15px;margin-bottom:6px}
.lede{font-size:18px;color:var(--soft)}
.note{font-size:15px;color:var(--soft)}
b{font-weight:700}
/* Loppet */
.lanes{list-style:none;padding:0;margin:22px 0 8px;display:grid;gap:14px}
.lane{background:var(--paper);border-radius:14px;padding:14px 16px 12px;border-left:6px solid var(--c)}
.lane-head{display:flex;align-items:baseline;gap:10px}
.place{font:800 22px/1 "Bricolage Grotesque",sans-serif;color:var(--c);min-width:18px}
.pname{font-weight:700;flex:1}
.pval{font:800 20px/1 "Bricolage Grotesque",sans-serif;font-variant-numeric:tabular-nums}
.track{position:relative;height:10px;background:var(--track);border-radius:5px;margin:12px 0 8px}
.startline{position:absolute;left:50%;top:-4px;bottom:-4px;width:2px;background:var(--soft);opacity:.5}
.runner{position:absolute;top:50%;width:20px;height:20px;margin:-10px 0 0 -10px;border-radius:50%;
background:var(--c);box-shadow:0 0 0 4px var(--paper);animation:run 1.1s cubic-bezier(.2,.8,.2,1) both}
@keyframes run{from{left:50%}}
@media (prefers-reduced-motion:reduce){.runner{animation:none}}
.lane-foot{display:flex;gap:12px;font-size:14px;color:var(--soft);align-items:baseline}
.lane-foot .up,.lane-foot .down{font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap}
.up{color:var(--up)}.down{color:var(--down)}
.pace{font-size:15px;color:var(--soft);margin-top:12px}
.chart-wrap{position:relative;overflow:hidden;margin:18px 0 6px;background:var(--paper);border-radius:14px;padding:12px}
.hidden{display:none}
/* Ljus */
.lights{list-style:none;padding:0;margin:0 0 14px;display:grid;gap:10px}
.light{display:flex;gap:12px;align-items:flex-start;background:var(--paper);border-radius:12px;padding:12px 14px;font-size:15px}
.light .dot{flex:none;width:16px;height:16px;border-radius:50%;margin-top:3px}
.go .dot{background:var(--up);box-shadow:0 0 0 4px color-mix(in srgb,var(--up) 22%,transparent)}
.stop .dot{background:var(--down);box-shadow:0 0 0 4px color-mix(in srgb,var(--down) 22%,transparent)}
/* Favoriter */
.bars{list-style:none;padding:0;margin:0;display:grid;gap:2px}
.bars li{display:grid;grid-template-columns:78px 1fr 48px;grid-template-areas:"t m n" "w w g";
align-items:center;column-gap:10px;padding:8px 0;border-bottom:1px solid var(--line)}
.tick{grid-area:t;font-weight:700}
.meter{grid-area:m;height:8px;background:var(--track);border-radius:4px;position:relative;overflow:hidden}
.meter::after{content:"";position:absolute;left:50%;top:0;bottom:0;width:2px;background:var(--bg)}
.meter span{display:block;height:100%;width:var(--p);background:var(--ai);border-radius:4px}
.num{grid-area:n;text-align:right;font-weight:700;font-variant-numeric:tabular-nums}
.w{grid-area:w;font-size:13px;color:var(--soft)}
.tag{grid-area:g;justify-self:end;white-space:nowrap;font-size:12px;padding:1px 8px;border-radius:99px}
.tag.pos{background:color-mix(in srgb,var(--up) 15%,transparent);color:var(--up)}
.tag.neg{background:color-mix(in srgb,var(--down) 15%,transparent);color:var(--down)}
details{margin-top:14px}
summary{cursor:pointer;font-weight:700;color:var(--ai);padding:6px 0}
summary:focus-visible,a:focus-visible{outline:3px solid var(--ai);outline-offset:3px;border-radius:4px}
/* Nyheter */
.news{list-style:none;padding:0;margin:0;display:grid;gap:10px}
.news li{background:var(--paper);border-radius:12px;padding:12px 14px;font-size:15px;display:grid;gap:4px;
border-left:4px solid var(--soft)}
.news li.pos{border-left-color:var(--up)}.news li.neg{border-left-color:var(--down)}
.sent{font-size:13px;color:var(--soft)}
.news li.pos .sent{color:var(--up)}.news li.neg .sent{color:var(--down)}
.why{font-size:14px;color:var(--soft)}
/* Facit */
.lab{list-style:none;padding:0;margin:14px 0;display:grid;gap:10px}
.lab li{background:var(--paper);border-radius:12px;padding:12px 14px}
.lab-head{display:flex;justify-content:space-between;gap:10px;align-items:center}
.lab .tag{grid-area:auto}
.lab-q{color:var(--soft);font-size:15px;margin:6px 0}.lab-a{font-size:15px;margin:0 0 4px}
.lab small{color:var(--soft)}
.big{font:800 64px/1 "Bricolage Grotesque",sans-serif;margin:8px 0}
/* Tabeller */
table{border-collapse:collapse;width:100%;font-size:15px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line)}
thead th{font-size:13px;color:var(--soft);font-weight:400}
td.r{text-align:right}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px}
.scroll{overflow-x:auto}
table.crisis th,table.crisis td{white-space:nowrap;font-size:14px}
td.na{color:var(--soft)}
.buy{color:var(--up)}.sell{color:var(--down)}
.glossary dt{font-weight:700;margin-top:12px}.glossary dd{margin:2px 0 0;color:var(--soft);font-size:15px}
.stale{background:var(--down);color:#fff;padding:14px 20px;font-size:15px}
.stale b{display:block}
footer{max-width:680px;margin:auto;padding:0 20px 40px;font-size:14px;color:var(--soft)}
</style></head>
<body>
<header class="top"><div class="top-in"><span class="brand">Tradingbot</span>
<small>Låtsaspengar, uppdaterad {{UPDATED}}</small></div></header>
<div id="stale" class="stale" hidden role="alert"></div>
<main>
{{SECTIONS}}
</main>
<footer>Ingen riktig handel sker. Historiska resultat är ingen garanti för framtiden.</footer>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
// Varna om boten inte har uppdaterats efter senaste vardagskörning (22:40, efter att USA stängt)
(function () {
  const upd = new Date("{{UPDATED_ISO}}");
  const now = new Date();
  const exp = new Date(now); exp.setHours(23, 30, 0, 0);
  if (now < exp) exp.setDate(exp.getDate() - 1);
  while (exp.getDay() === 0 || exp.getDay() === 6) exp.setDate(exp.getDate() - 1);
  exp.setHours(22, 0, 0, 0);
  if (upd < exp) {
    const el = document.getElementById('stale');
    el.hidden = false;
    el.innerHTML = '<b>Boten har inte uppdaterats sedan ' + upd.toLocaleString('sv-SE', {weekday: 'long', day: 'numeric', month: 'long', hour: '2-digit', minute: '2-digit'}) +
      '.</b> Den dagliga körningen kan ha misslyckats. Titta under Issues på GitHub eller fråga Claude.';
  }
})();
const D = {{DATA}};
if (window.Chart) {
  const css = getComputedStyle(document.documentElement);
  const soft = css.getPropertyValue('--soft').trim(), line = css.getPropertyValue('--line').trim();
  Chart.defaults.font.family = '"Figtree", system-ui, sans-serif';
  Chart.defaults.color = soft;
  const kr = v => Math.round(v).toLocaleString('sv-SE') + ' kr';
  const base = (extra = {}) => ({responsive: true, maintainAspectRatio: false, animation: false,
    interaction: {mode: 'index', intersect: false},
    plugins: {legend: {position: 'bottom', labels: {boxWidth: 10, boxHeight: 10, usePointStyle: true}}},
    scales: {x: {grid: {display: false}, ticks: {maxTicksLimit: 6}},
             y: {grid: {color: line}, border: {display: false}}}, ...extra});
  const fit = id => { const c = document.getElementById(id); if (c) c.parentElement.style.height = c.getAttribute('height') + 'px'; return c; };
  if (D.race && D.race.labels.length > 1 && fit('raceChart')) {
    const ds = D.race.series.map(s => ({label: s.namn, data: s.data, borderColor: s.farg, backgroundColor: s.farg,
      borderWidth: s.namn === 'Gör ingenting' ? 2 : 3, borderDash: s.namn === 'Gör ingenting' ? [2, 3] : [],
      pointRadius: 0, tension: .25, spanGaps: true}));
    if (D.race.spy.length) ds.push({label: D.race.spy_namn, data: D.race.spy, borderColor: soft, borderDash: [5, 4],
      borderWidth: 1.5, pointRadius: 0});
    const o = base(); o.scales.y.ticks = {callback: kr};
    o.plugins.tooltip = {callbacks: {label: c => c.dataset.label + ': ' + kr(c.parsed.y)}};
    new Chart(document.getElementById('raceChart'), {type: 'line', data: {labels: D.race.labels, datasets: ds}, options: o});
  }
  if (D.bt && fit('btChart')) {
    const ds = D.bt.series.map((s, i) => ({label: s.namn, data: s.data, borderColor: s.farg, backgroundColor: s.farg,
      borderWidth: i === 3 ? 1.5 : 2.5, pointRadius: 0, tension: .2}));
    const o = base(); o.scales.y.type = 'logarithmic';
    o.scales.y.ticks = {callback: v => [100000, 200000, 400000, 800000].includes(v) ? Math.round(v / 1000) + ' tkr' : ''};
    o.scales.x.ticks = {maxTicksLimit: 7, callback: function(v) { return this.getLabelForValue(v).slice(0, 4); }};
    o.plugins.tooltip = {callbacks: {label: c => c.dataset.label + ': ' + kr(c.parsed.y)}};
    new Chart(document.getElementById('btChart'), {type: 'line', data: {labels: D.bt.labels, datasets: ds}, options: o});
  }
  if (D.auc && fit('aucChart')) {
    const up = css.getPropertyValue('--up').trim(), down = css.getPropertyValue('--down').trim();
    const o = base({plugins: {legend: {display: false}, tooltip: {callbacks: {label: c => 'Träffsäkerhet ' + (D.auc.data[c.dataIndex] * 100).toFixed(1) + ' %'}}}});
    const m = Math.max(...D.auc.data.map(a => Math.abs(a - .5))) + .005; o.scales.y.min = -m; o.scales.y.max = m;
    o.scales.y.ticks = {callback: v => Math.abs(v) < 1e-9 ? 'slumpen' : (Math.abs(Math.abs(v) - m) < 1e-9 ? (v > 0 ? 'bättre' : 'sämre') : ''), stepSize: m};
    o.scales.x.ticks = {maxTicksLimit: 7, callback: function(v) { return this.getLabelForValue(v).slice(0, 4); }};
    new Chart(document.getElementById('aucChart'), {type: 'bar', data: {labels: D.auc.labels, datasets: [{
      data: D.auc.data.map(a => a - 0.5), backgroundColor: D.auc.data.map(a => a > .5 ? up : down), borderRadius: 3}]}, options: o});
  }
  if (D.risk && fit('riskChart')) {
    const o = base(); o.scales.y.ticks = {callback: v => v + ' %'};
    o.scales.x.ticks = {maxTicksLimit: 7, callback: function(v) { return this.getLabelForValue(v).slice(0, 4); }};
    new Chart(document.getElementById('riskChart'), {type: 'line', data: {labels: D.risk.labels, datasets: [
      {label: 'AI-riskmodellen', data: D.risk.ai, borderColor: '#e67700', backgroundColor: '#e67700', borderWidth: 2.5, pointRadius: 0, tension: .3},
      {label: 'Enkel gissning', data: D.risk.enkel, borderColor: soft, backgroundColor: soft, borderWidth: 1.5, borderDash: [5, 4], pointRadius: 0, tension: .3}]}, options: o});
  }
}
</script>
</body></html>"""


def backtest_current() -> bool:
    try:
        return json.load(open(F_BACKTEST)).get("version") == VERSION
    except Exception:
        return False


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "allt"
    if cmd == "backtest":
        cmd_backtest()
    elif cmd == "papper":
        cmd_papper()
    elif cmd == "rapport":
        cmd_rapport()
    elif cmd == "allt":
        if not backtest_current():
            cmd_backtest()
        cmd_papper()
        cmd_rapport()
    else:
        print(__doc__)
