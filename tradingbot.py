"""
AI-tradingbot v2 – två låtsasportföljer sida vid sida.

  AI-portföljen:   äger alla aktier, men lägger MER pengar i de aktier AI-modellen
                   (och nyheterna) tror mest på, och mindre i de den tror minst på.
  Enkla portföljen: lika mycket i alla aktier. Ingen AI.
  Båda:            halverar insatsen i en marknad (USA/Sverige) när marknaden ligger
                   under sitt 200-dagars snitt. Ombalanserar en gång i månaden.

Varför: i testerna 2014–idag gav spridning + marknadsfilter bäst riskjusterad
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
START = "2010-01-01"
HORIZON = 20            # förutsäg ca 1 månad framåt
COST = 0.0015           # courtage + spread
REBALANCE_DAYS = 20     # ombalansera ungefär en gång i månaden
FILTER_FACTOR = 0.5     # insats när marknaden är under 200-dagars snitt
TILT = 1.0              # hur mycket AI:n får flytta vikterna (0 = ingen AI)
NEWS_WEIGHT = 0.05
NEWS_DAYS = 7
START_CASH = 100_000
MIN_TRADE = 0.005       # handla inte om skillnaden är under 0,5 % av portföljen
CLAUDE_MODEL = "claude-haiku-5-5"
VERSION = 3

DIR = os.path.dirname(os.path.abspath(__file__))
F_PORTFOLIO = os.path.join(DIR, "papper_portfolj.json")
F_TRADES = os.path.join(DIR, "papper_affarer.csv")
F_HISTORY = os.path.join(DIR, "papper_historik.csv")
F_PRED = os.path.join(DIR, "prediktioner.csv")
F_BACKTEST = os.path.join(DIR, "backtest.json")
F_REPORT = os.path.join(DIR, "rapport.html")
PORTS = {"ai": "AI-portföljen", "enkel": "Enkla portföljen", "risk": "Risk-portföljen"}
F_NEWS = os.path.join(DIR, "nyheter_arkiv.csv")
MACRO = ["vix", "vix_chg20", "ranta10", "ranta_chg60", "usdsek_chg60"]

MODEL_PARAMS = dict(max_iter=300, learning_rate=0.03, max_depth=4,
                    min_samples_leaf=200, l2_regularization=1.0, random_state=42)


# ---------------- Data ----------------
_cache: dict = {}


def load_data(ticker: str) -> pd.DataFrame:
    if ticker in _cache:
        return _cache[ticker]
    import yfinance as yf
    df = yf.download(ticker, start=START, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    _cache[ticker] = df
    return df


def market_of(t: str) -> str:
    return MARKET_SE if t.endswith(".ST") else MARKET_US


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
        f["fwd"] = df["Close"].shift(-HORIZON) / df["Close"] - 1
        f["ret1"] = df["Close"].pct_change().shift(-1)
        f["pris"] = df["Close"]
        f["aktie"] = t
        f["se"] = int(t.endswith(".ST"))
        frames.append(f)
    p = pd.concat(frames)
    p.index.name = "Date"
    p = p.reset_index().dropna(subset=["avst_ma200", "fran_arshogsta"])
    for c in RANK_COLS:
        p[c + "_rank"] = p.groupby("Date")[c].rank(pct=True)
    # Mål: slår aktien mitten-aktien den kommande månaden?
    med = p.groupby("Date")["fwd"].transform("median")
    p["y"] = (p["fwd"] > med).astype(float).where(p["fwd"].notna())
    feats = [c for c in p.columns if c not in ("Date", "fwd", "ret1", "pris", "aktie", "y")]
    # Makrodata (används av riskmodellen)
    vix, tnx = load_data("^VIX")["Close"], load_data("^TNX")["Close"]
    fx = load_data("USDSEK=X")["Close"]
    macro = pd.DataFrame({"vix": vix, "vix_chg20": vix.pct_change(20), "ranta10": tnx,
                          "ranta_chg60": tnx.diff(60), "usdsek_chg60": fx.pct_change(60)}).ffill()
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


# ---------------- Portföljvikter ----------------
def target_weights(scores: pd.Series, below: dict, mode: str, pred_vol=None) -> pd.Series:
    """mode: 'ai' (mer i aktier AI:n tror på), 'enkel' (lika), 'risk' (mindre i stormiga aktier).
    below: {'us': bool, 'se': bool} – marknaden under sitt 200-dagars snitt."""
    if mode == "ai":
        z = (scores.rank(pct=True) - 0.5) * 2 * TILT
        w = (1 + z).clip(lower=0)
    elif mode == "risk" and pred_vol is not None:
        w = 1 / pred_vol.reindex(scores.index).fillna(pred_vol.median()).clip(lower=0.05)
    else:
        w = pd.Series(1.0, index=scores.index)
    w = w / w.sum()
    for t in w.index:
        if below["se" if t.endswith(".ST") else "us"]:
            w[t] *= FILTER_FACTOR
    return w


# ---------------- Backtest ----------------
def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    years = len(r) / 252
    return {"arlig": float(eq.iloc[-1] ** (1 / years) - 1),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0,
            "max_ras": float((eq / eq.cummax() - 1).min())}


def cmd_backtest():
    print("Bygger data för alla aktier...")
    p, feats = build_panel()
    p["prob"] = np.nan
    p["pred_vol"] = np.nan
    rf = risk_feats(feats)
    learning = []
    start = pd.Timestamp("2014-01-01")
    while start < p["Date"].max():
        end = start + pd.DateOffset(months=6)
        tr = p[p["Date"] < start - pd.Timedelta(days=35)].dropna(subset=["y"])
        te = (p["Date"] >= start) & (p["Date"] < end)
        if te.any():
            m = new_model().fit(tr[feats], tr["y"])
            p.loc[te, "prob"] = m.predict_proba(p.loc[te, feats])[:, 1]
            trv = p[p["Date"] < start - pd.Timedelta(days=35)].dropna(subset=["log_fwd_vol"])
            mv = new_risk_model().fit(trv[rf], trv["log_fwd_vol"])
            p.loc[te, "pred_vol"] = np.exp(mv.predict(p.loc[te, rf]))
            sc = p.loc[te].dropna(subset=["y"])
            if len(sc) > 200 and sc["y"].nunique() == 2:
                sv = p.loc[te].dropna(subset=["fwd_vol"])
                learning.append({"period": start.strftime("%Y-%m"),
                                 "auc": float(roc_auc_score(sc["y"], sc["prob"])),
                                 "risk_fel_ai": float((sv["pred_vol"] - sv["fwd_vol"]).abs().mean()),
                                 "risk_fel_enkel": float((sv["vol_20"] * np.sqrt(252) - sv["fwd_vol"]).abs().mean()),
                                 "traningsrader": len(tr)})
                print(f"   {start:%Y-%m}: AUC {learning[-1]['auc']:.3f}")
        start = end

    q = p.dropna(subset=["prob"])
    R = q.pivot(index="Date", columns="aktie", values="ret1").fillna(0)
    P = q.pivot(index="Date", columns="aktie", values="prob")
    reg = q.pivot(index="Date", columns="aktie", values="marknad_avst_ma200")
    PV = q.pivot(index="Date", columns="aktie", values="pred_vol")

    def run(mode, use_filter=True):
        W = pd.DataFrame(0.0, index=R.index, columns=R.columns)
        w, last_below = None, None
        for i, d in enumerate(R.index):
            r_d = reg.loc[d]
            below = {"us": bool((r_d[[c for c in r_d.index if not c.endswith(".ST")]] < 0).mean() > 0.5),
                     "se": bool((r_d[[c for c in r_d.index if c.endswith(".ST")]] < 0).mean() > 0.5)}
            if not use_filter:
                below = {"us": False, "se": False}
            if w is None or i % REBALANCE_DAYS == 0 or below != last_below:
                s = P.loc[d].dropna()
                if len(s) > 10:
                    w = target_weights(s, below, mode, PV.loc[d]).reindex(R.columns).fillna(0)
                    last_below = below
            if w is not None:
                W.loc[d] = w
        gross = (W * R).sum(axis=1)
        turn = W.diff().abs().sum(axis=1).fillna(0)
        return gross - turn * COST

    res = {"AI-portföljen": run("ai"), "Enkla portföljen": run("enkel"),
           "Risk-portföljen": run("risk"), "Lika vikt utan filter": run("enkel", use_filter=False)}
    out = {"version": VERSION, "skapad": datetime.now().isoformat(timespec="minutes"),
           "larande": learning, "strategier": {}, "kurvor": {}}
    for name, r in res.items():
        out["strategier"][name] = {"hela": stats(r), "fore_2022": stats(r.loc[:"2021-12-31"]),
                                   "fran_2022": stats(r.loc["2022-01-01":])}
        wk = (1 + r).cumprod().resample("W").last()
        out["kurvor"][name] = wk.round(4).tolist()
        out["kurvor"]["datum"] = [d.strftime("%Y-%m-%d") for d in wk.index]
        s = out["strategier"][name]["hela"]
        print(f"{name:22} {s['arlig']:+.1%}/år  Sharpe {s['sharpe']:.2f}  ras {s['max_ras']:.0%}")
    json.dump(out, open(F_BACKTEST, "w"), ensure_ascii=False, indent=1)


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
        if pf.get("version") == 2:
            # Behåll AI- och Enkla portföljen, lägg till Risk-portföljen
            for k in PORTS:
                pf["portfoljer"].setdefault(k, {"kassa": START_CASH, "innehav": {},
                                                "senast_ombalans": None, "senaste_filter": None})
            pf["version"] = VERSION
            print("Lägger till Risk-portföljen.")
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
    evaluate_predictions(panel)

    # Träna AI-modellen på all historik med kända utfall
    train = panel.dropna(subset=["y"])
    model = new_model().fit(train[feats], train["y"])
    last = panel.sort_values("Date").groupby("aktie").tail(1).set_index("aktie")
    last["ml"] = model.predict_proba(last[feats])[:, 1]
    sample = train[train["Date"] > train["Date"].max() - pd.Timedelta(days=365)]
    sample = sample.sample(min(4000, len(sample)), random_state=0)
    imp = permutation_importance(model, sample[feats], sample["y"], n_repeats=3,
                                 random_state=0, scoring="roc_auc")
    top_feats = sorted(zip(feats, imp.importances_mean), key=lambda z: -z[1])[:8]
    # Riskmodellen: hur mycket väntas varje aktie svänga kommande månad?
    rf = risk_feats(feats)
    trv = panel.dropna(subset=["log_fwd_vol"])
    last_r = panel.sort_values("Date").groupby("aktie").tail(1).set_index("aktie")
    pred_vol = pd.Series(np.exp(new_risk_model().fit(trv[rf], trv["log_fwd_vol"]).predict(last_r[rf])),
                         index=last_r.index)

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
                     "risk": float(pred_vol.get(t, np.nan)), "nyhetslista": items[:4]})
    # Nyhetsarkiv – samlas för framtida träning
    if archive:
        new = pd.DataFrame(archive)
        if os.path.exists(F_NEWS):
            old = pd.read_csv(F_NEWS)
            new = new[~(new["aktie"] + new["rubrik"]).isin(old["aktie"] + old["rubrik"])]
        append_csv(F_NEWS, new.to_dict("records"))
    scores = pd.Series({s["aktie"]: s["poang"] for s in sigs})
    prices = {s["aktie"]: to_sek(s["aktie"], s["pris"], fx) for s in sigs}

    mk = {}
    for key, tick in (("us", MARKET_US), ("se", MARKET_SE)):
        c = load_data(tick)["Close"]
        mk[key] = float(c.iloc[-1] / c.rolling(200).mean().iloc[-1] - 1)
    below = {k: v < 0 for k, v in mk.items()}

    all_trades, hist_row = [], {"datum": today}
    for key, label in PORTS.items():
        port = pf["portfoljer"][key]
        hold = {t: float(n) for t, n in port["innehav"].items()}
        equity = port["kassa"] + sum(n * prices.get(t, 0) for t, n in hold.items())
        last_reb = port["senast_ombalans"]
        due = (last_reb is None or (pd.Timestamp(today) - pd.Timestamp(last_reb)).days >= 28
               or port["senaste_filter"] != below)
        if due:
            w = target_weights(scores, below, key, pred_vol)
            for t in sorted(w.index, key=lambda t: w[t] * equity - hold.get(t, 0) * prices[t]):
                diff = w[t] * equity - hold.get(t, 0) * prices[t]
                if abs(diff) < MIN_TRADE * equity:
                    continue
                if diff > 0:
                    diff = min(diff, port["kassa"] / (1 + COST))
                    if diff <= 0:
                        continue
                n = diff / prices[t]
                hold[t] = hold.get(t, 0) + n
                port["kassa"] -= diff + abs(diff) * COST
                all_trades.append({"datum": today, "portfolj": label, "aktie": t,
                                   "handling": "KÖP" if diff > 0 else "SÄLJ",
                                   "belopp_sek": round(abs(diff), 2), "vikt": round(float(w[t]), 4)})
            hold = {t: n for t, n in hold.items() if n * prices.get(t, 0) > 1}
            port["senast_ombalans"], port["senaste_filter"] = today, below
            port["vikter"] = {t: round(float(v), 4) for t, v in w.items()}
        port["innehav"] = hold
        equity = port["kassa"] + sum(n * prices.get(t, 0) for t, n in hold.items())
        hist_row[key] = round(equity, 2)
        print(f"   {label}: {equity:,.0f} kr ({'ombalanserad' if due else 'oförändrad'})")

    spy = prices.get("SPY") or to_sek("SPY", float(load_data("SPY")["Close"].iloc[-1]), fx)
    pf.setdefault("spy_start", spy)
    hist_row["jamforelse_spy"] = round(START_CASH * spy / pf["spy_start"], 2)

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


# ---------------- Rapport ----------------
def fig_to_img(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pct(x, sign=True):
    return f"{x:+.1%}" if sign else f"{x:.0%}"


def cmd_rapport():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"ai": "#2563eb", "enkel": "#16a34a", "risk": "#f59e0b", "jamforelse_spy": "#9ca3af"}
    parts = []

    if os.path.exists(F_PORTFOLIO) and os.path.exists(F_HISTORY):
        pf = json.load(open(F_PORTFOLIO))
        h = pd.read_csv(F_HISTORY)
        lastrow = h.iloc[-1]
        cards = ""
        for key, label in PORTS.items():
            v = lastrow.get(key, np.nan)
            if pd.isna(v):
                continue
            d = v / START_CASH - 1
            cards += (f"<div class='card'><small>{label}</small><b>{v:,.0f} kr</b>"
                      f"<span class='{'pos' if d >= 0 else 'neg'}'>{d:+.2%}</span></div>")
        cards += (f"<div class='card'><small>Om allt låg i SPY</small><b>{lastrow['jamforelse_spy']:,.0f} kr</b>"
                  f"<span>{lastrow['jamforelse_spy'] / START_CASH - 1:+.2%}</span></div>")
        parts.append(f"<h2>Låtsaspengar</h2><p class='note'>Start {pf['start']} med 100 000 kr i "
                     f"varje portfölj.</p><div class='cards'>{cards}</div>")
        if len(h) > 1:
            fig, ax = plt.subplots(figsize=(7, 2.6))
            dts = pd.to_datetime(h["datum"])
            for k, lab in list(PORTS.items()) + [("jamforelse_spy", "SPY")]:
                if k not in h:
                    continue
                ax.plot(dts, h[k], label=lab, color=colors[k], lw=2 if k != "jamforelse_spy" else 1.3)
            ax.axhline(START_CASH, color="#e5e7eb", lw=1)
            ax.legend(frameon=False)
            parts.append(fig_to_img(fig))
        mk = pf.get("marknad", {})
        if mk:
            def mtxt(k, n):
                v = mk[k]
                return (f"<li><b>{n}</b>: {v:+.1%} mot 200-dagars snitt – "
                        + ("<span class='neg'>under, halv insats</span>" if v < 0
                           else "<span class='pos'>över, full insats</span>") + "</li>")
            parts.append(f"<h2>Marknadsfiltret</h2><ul>{mtxt('us', 'USA (SPY)')}{mtxt('se', 'Sverige (OMX)')}</ul>")
        sig = sorted(pf.get("signaler", []), key=lambda s: -s["poang"])
        if sig:
            vik = pf["portfoljer"]["ai"].get("vikter", {})
            vr = pf["portfoljer"].get("risk", {}).get("vikter", {})
            rows = ""
            for s in sig:
                risk = f"{s['risk']:.0%}" if s.get("risk") == s.get("risk") and s.get("risk") else "–"
                rows += (f"<tr><td>{s['aktie']}</td><td>{s['ml']:.0%}</td><td>{s['nyheter']:+.2f}</td>"
                         f"<td><b>{s['poang']:.0%}</b></td><td>{vik.get(s['aktie'], 0):.1%}</td>"
                         f"<td>{risk}</td><td>{vr.get(s['aktie'], 0):.1%}</td></tr>")
            parts.append("<h2>AI:ns bedömning idag</h2><p class='note'>Poäng = chans att slå mitten-aktien "
                         "nästa månad (styr AI-portföljen). Svängning = hur mycket riskmodellen tror att "
                         "aktien rör sig per år (styr Risk-portföljen: stormigare aktie får mindre).</p>"
                         "<table><tr><th>Aktie</th><th>ML</th><th>Nyh.</th><th>Poäng</th>"
                         f"<th>Vikt AI</th><th>Sväng.</th><th>Vikt Risk</th></tr>{rows}</table>")
            news = ""
            for s in sig:
                for n in s.get("nyhetslista", [])[:2]:
                    cls = "pos" if n["poang"] > 0.1 else "neg" if n["poang"] < -0.1 else ""
                    rel = "" if n.get("relevant", True) else " <i>(fel bolag – räknas ej)</i>"
                    why = f"<br><small>{esc(n['motivering'])}</small>" if n.get("motivering") else ""
                    news += (f"<li><span class='{cls}'>{n['poang']:+.2f}</span> <b>{s['aktie']}</b> "
                             f"{esc(n['rubrik'])}{rel}{why}</li>")
            if news:
                metod = next((n["metod"] for s in sig for n in s.get("nyhetslista", [])), "")
                parts.append(f"<h2>Nyheter boten läst ({metod})</h2><ul class='news'>{news}</ul>")
        if pf.get("viktigast"):
            parts.append("<h2>Vad AI:n tittar mest på</h2><p>" +
                         ", ".join(n for n, _ in pf["viktigast"]) + "</p>")
    if os.path.exists(F_TRADES):
        tr = pd.read_csv(F_TRADES)
        if "portfolj" in tr:
            tr = tr.tail(20).iloc[::-1]
            rows = "".join(f"<tr><td>{r.datum}</td><td>{r.portfolj.split('-')[0].split()[0]}</td>"
                           f"<td>{r.handling}</td><td>{r.aktie}</td><td>{r.belopp_sek:,.0f} kr</td></tr>"
                           for r in tr.itertuples())
            parts.append(f"<h2>Senaste affärer</h2><table>{rows}</table>")
    if os.path.exists(F_PRED):
        p = pd.read_csv(F_PRED)
        if "ratt_ml" in p and p["ratt_ml"].notna().any():
            d = p.dropna(subset=["ratt_ml"])
            parts.append(f"<h2>Facit i verkligheten</h2><p>{len(d)} gissningar har fått facit. "
                         f"AI rätt: <b>{d['ratt_ml'].mean():.0%}</b> · med nyheter: "
                         f"<b>{d['ratt_med_nyheter'].mean():.0%}</b> (50 % = slumpen)</p>")
        else:
            parts.append(f"<h2>Facit i verkligheten</h2><p class='note'>Första facit kommer efter "
                         f"{HORIZON} handelsdagar (ungefär en månad).</p>")

    if os.path.exists(F_BACKTEST):
        bt = json.load(open(F_BACKTEST))
        if bt.get("version") == VERSION:
            L = pd.DataFrame(bt["larande"])
            fig, ax = plt.subplots(figsize=(7, 2.4))
            ax.bar(pd.to_datetime(L["period"]), L["auc"] - 0.5, bottom=0.5, width=150,
                   color=["#16a34a" if a > 0.5 else "#dc2626" for a in L["auc"]])
            ax.axhline(0.5, color="#111", lw=0.8)
            ax.set_ylim(0.44, 0.58)
            ax.set_ylabel("AUC")
            parts.append("<h2>Hur AI:n lärt sig genom åren</h2><p class='note'>Varje stapel är ett "
                         "halvår den aldrig sett. Grönt = bättre än slumpen (0,5), rött = sämre.</p>"
                         + fig_to_img(fig))
            if "risk_fel_ai" in L:
                fig, ax = plt.subplots(figsize=(7, 2.2))
                dts_l = pd.to_datetime(L["period"])
                ax.plot(dts_l, L["risk_fel_enkel"], color="#9ca3af", lw=1.5, label="Enkel gissning")
                ax.plot(dts_l, L["risk_fel_ai"], color="#f59e0b", lw=2, label="AI-riskmodellen")
                ax.set_ylabel("Fel (lägre = bättre)")
                ax.legend(frameon=False)
                parts.append("<h2>Hur riskmodellen lärt sig</h2><p class='note'>Hur fel den gissar "
                             "aktiernas svängningar nästa månad, jämfört med att bara anta att de svänger "
                             "som förra månaden.</p>" + fig_to_img(fig))
            fig, ax = plt.subplots(figsize=(7, 2.6))
            dts = pd.to_datetime(bt["kurvor"]["datum"])
            for name, col in (("AI-portföljen", "#2563eb"), ("Enkla portföljen", "#16a34a"),
                              ("Risk-portföljen", "#f59e0b"),
                              ("Lika vikt utan filter", "#9ca3af")):
                ax.plot(dts, bt["kurvor"][name], label=name, color=col, lw=1.6)
            ax.set_yscale("log")
            ax.legend(frameon=False)
            rows = "".join(
                f"<tr><td>{n}</td><td>{s['fore_2022']['arlig']:+.1%}</td><td>{s['fran_2022']['arlig']:+.1%}</td>"
                f"<td>{s['hela']['sharpe']:.2f}</td><td>{s['hela']['max_ras']:.0%}</td></tr>"
                for n, s in bt["strategier"].items())
            parts.append("<h2>Backtest 2014–idag</h2>" + fig_to_img(fig) +
                         "<table><tr><th>Strategi</th><th>Per år t.o.m. 2021</th><th>Per år 2022–</th>"
                         f"<th>Sharpe</th><th>Värsta ras</th></tr>{rows}</table>"
                         "<p class='note'>Aktierna är dagens stora bolag, vilket gör alla historiska "
                         "siffror lite för bra (de som gick dåligt finns inte med).</p>")

    html = f"""<!DOCTYPE html><html lang="sv"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Tradingbot</title>
<meta http-equiv="refresh" content="1800"><link rel="manifest" href="manifest.json">
<meta name="apple-mobile-web-app-capable" content="yes"><meta name="theme-color" content="#2563eb">
<link rel="apple-touch-icon" href="ikon.png">
<style>body{{font-family:system-ui,sans-serif;max-width:760px;margin:auto;padding:16px;color:#111;background:#fafafa}}
h1{{font-size:22px}}h2{{font-size:16px;margin-top:28px;border-bottom:1px solid #e5e7eb;padding-bottom:4px}}
img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{padding:5px 6px;border-bottom:1px solid #eee;text-align:left}}
.cards{{display:flex;gap:10px;flex-wrap:wrap}}.card{{flex:1;min-width:150px;background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:10px}}
.card small{{display:block;color:#666}}.card b{{display:block;font-size:22px;margin:2px 0}}
.pos{{color:#16a34a;font-weight:600}}.neg{{color:#dc2626;font-weight:600}}
.note{{color:#666;font-size:13px}}.news li{{margin-bottom:8px;font-size:13px}}
.warn{{background:#fff7ed;border:1px solid #fed7aa;padding:10px;border-radius:8px;font-size:13px}}</style></head>
<body><h1>Tradingbot</h1><p class="note">Uppdaterad {datetime.now():%Y-%m-%d %H:%M}</p>
<div class="warn">Låtsaspengar. Historiskt resultat är ingen garanti för framtiden.</div>
{''.join(parts)}</body></html>"""
    open(F_REPORT, "w").write(html)
    docs = os.path.join(DIR, "docs")
    os.makedirs(docs, exist_ok=True)
    open(os.path.join(docs, "index.html"), "w").write(html)
    json.dump({"name": "Tradingbot", "short_name": "Tradingbot", "start_url": ".",
               "display": "standalone", "background_color": "#fafafa", "theme_color": "#2563eb",
               "icons": [{"src": "ikon.png", "sizes": "512x512", "type": "image/png"}]},
              open(os.path.join(docs, "manifest.json"), "w"))
    icon = os.path.join(docs, "ikon.png")
    if not os.path.exists(icon):
        fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.plot([0.12, 0.32, 0.48, 0.62, 0.88], [0.25, 0.5, 0.4, 0.65, 0.8], color="white",
                lw=22, solid_capstyle="round", solid_joinstyle="round")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        fig.patch.set_facecolor("#2563eb")
        fig.savefig(icon); plt.close(fig)
    print("Rapport sparad: rapport.html + docs/")


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
