"""
AI-tradingbot med nyhetsanalys och papperstrading (låtsaspengar).

KOMMANDON
    python tradingbot.py backtest   # tränar och testar på historik (tar några minuter)
    python tradingbot.py papper     # en handelsdag med låtsaspengar
    python tradingbot.py rapport    # bygger rapport.html
    python tradingbot.py allt       # papper + rapport (kör varje vardag)

INSTALLERA
    pip install yfinance pandas numpy scikit-learn matplotlib vaderSentiment anthropic

NYHETSANALYS
    Om ANTHROPIC_API_KEY är satt läser Claude nyheterna, bedömer om de handlar om
    rätt bolag och om de är positiva eller negativa. Annars används VADER, en
    enklare ordlistemetod.

ÄRLIGT
    ML-delen är testad på historik och slog inte köp-och-behåll i snitt.
    Nyhetsdelen går inte att backtesta (gamla nyheter finns inte gratis), därför
    loggas allt i prediktioner.csv så att vi ser i efterhand om den hjälper.
    Ingen garanti för vinst.
"""
import json
import os
import sys
import base64
import io
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ---------------- Inställningar ----------------
TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "VOLV-B.ST", "ERIC-B.ST", "INVE-B.ST", "ATCO-A.ST"]
MARKET = "SPY"
START = "2010-01-01"
HORIZON = 20           # förutsäg 20 handelsdagar (ca 1 månad) framåt
COST = 0.0015          # courtage + spread per affär (0,15 %)
TRAIN_YEARS = 4
TEST_MONTHS = 6
BUY_THRESHOLD = 0.58   # gå in när poängen är minst 58 %
SELL_THRESHOLD = 0.45  # gå ur först under 45 %
SMOOTH = 10            # jämna ut ML-signalen över 10 dagar
NEWS_WEIGHT = 0.05     # nyheter kan flytta poängen max ±5 procentenheter
NEWS_DAYS = 7          # titta på nyheter från senaste veckan
START_CASH = 100_000   # låtsaspengar i kronor
CLAUDE_MODEL = "claude-haiku-5-5"

DIR = os.path.dirname(os.path.abspath(__file__))
F_PORTFOLIO = os.path.join(DIR, "papper_portfolj.json")
F_TRADES = os.path.join(DIR, "papper_affarer.csv")
F_HISTORY = os.path.join(DIR, "papper_historik.csv")
F_PRED = os.path.join(DIR, "prediktioner.csv")
F_BACKTEST = os.path.join(DIR, "backtest.json")
F_REPORT = os.path.join(DIR, "rapport.html")

MODEL_PARAMS = dict(max_iter=300, learning_rate=0.03, max_depth=4,
                    min_samples_leaf=50, l2_regularization=1.0, random_state=42)


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


def usdsek() -> float:
    df = load_data("USDSEK=X")
    return float(df["Close"].iloc[-1])


def to_sek(ticker: str, price: float, fx: float) -> float:
    return price if ticker.endswith(".ST") else price * fx


# ---------------- Egenskaper ----------------
def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n).mean()
    return 100 - 100 / (1 + up / down)


def make_features(df: pd.DataFrame, market: pd.Series | None) -> pd.DataFrame:
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
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - c.shift()).abs(),
                    (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c
    f["volym_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    f["fran_arshogsta"] = c / c.rolling(252).max() - 1
    f["veckodag"] = df.index.dayofweek
    f["manad"] = df.index.month
    if market is not None:
        m = market.reindex(df.index).ffill()
        f["marknad_avk_20d"] = m.pct_change(20)
        f["marknad_avst_ma200"] = m / m.rolling(200).mean() - 1
        f["relativ_styrka_60d"] = c.pct_change(60) - m.pct_change(60)
    return f


def make_target(df: pd.DataFrame) -> pd.Series:
    fwd = df["Close"].shift(-HORIZON) / df["Close"] - 1
    return (fwd > COST * 2).astype(float).where(fwd.notna())


def dataset(ticker: str):
    df = load_data(ticker)
    market = load_data(MARKET)["Close"]
    X = make_features(df, market)
    y = make_target(df)
    data = X.join(y.rename("y")).dropna(subset=list(X.columns))
    return df, data, list(X.columns)


# ---------------- Walk-forward (så ser vi hur den lär sig) ----------------
def walk_forward(data: pd.DataFrame, cols: list):
    probs = pd.Series(np.nan, index=data.index)
    learning = []
    start = data.index[0] + pd.DateOffset(years=TRAIN_YEARS)
    while start < data.index[-1]:
        end = start + pd.DateOffset(months=TEST_MONTHS)
        train = data[data.index < start].iloc[:-HORIZON].dropna(subset=["y"])
        test = data[(data.index >= start) & (data.index < end)]
        if len(train) > 500 and len(test) > 0:
            model = HistGradientBoostingClassifier(
                early_stopping=True, validation_fraction=0.15, **MODEL_PARAMS)
            model.fit(train[cols], train["y"])
            p = model.predict_proba(test[cols])[:, 1]
            probs.loc[test.index] = p
            scored = test.dropna(subset=["y"])
            if len(scored) > 20 and scored["y"].nunique() == 2:
                ps = probs.loc[scored.index]
                learning.append({
                    "period": start.strftime("%Y-%m"),
                    "traningsdagar": len(train),
                    "auc": float(roc_auc_score(scored["y"], ps)),
                    "traffsakerhet": float(((ps >= 0.5) == scored["y"]).mean()),
                    "basniva": float(max(scored["y"].mean(), 1 - scored["y"].mean())),
                })
        start = end
    return probs.dropna(), learning


def positions_from_probs(probs: pd.Series) -> pd.Series:
    p = probs.rolling(SMOOTH, min_periods=1).mean()
    state, held = [], 0.0
    for v in p.values:
        if v >= BUY_THRESHOLD:
            held = 1.0
        elif v < SELL_THRESHOLD:
            held = 0.0
        state.append(held)
    return pd.Series(state, index=p.index)


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    years = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else 0
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    return {"total": float(eq.iloc[-1] - 1), "arlig": float(cagr),
            "sharpe": float(sharpe), "max_ras": float((eq / eq.cummax() - 1).min())}


def cmd_backtest():
    out = {"skapad": datetime.now().isoformat(timespec="minutes"), "aktier": {}}
    for t in TICKERS:
        try:
            df, data, cols = dataset(t)
        except Exception as e:
            print(f"{t}: kunde inte hämta data ({e})")
            continue
        print(f"Backtestar {t} ({len(df)} dagar)...")
        probs, learning = walk_forward(data, cols)
        ret = df["Close"].pct_change().reindex(probs.index).fillna(0)
        pos = positions_from_probs(probs).shift(1).fillna(0)
        trades = pos.diff().abs().fillna(pos.abs())
        strat = pos * ret - trades * COST
        eq = (1 + strat).cumprod()
        bh = (1 + ret).cumprod()
        weekly = pd.DataFrame({"bot": eq, "behall": bh}).resample("W").last()
        out["aktier"][t] = {
            "bot": stats(strat), "behall": stats(ret),
            "tid_i_marknad": float((pos > 0).mean()),
            "affarer": int((trades > 0).sum()),
            "larande": learning,
            "kurva": {"datum": [d.strftime("%Y-%m-%d") for d in weekly.index],
                      "bot": weekly["bot"].round(4).tolist(),
                      "behall": weekly["behall"].round(4).tolist()},
        }
        b, h = out["aktier"][t]["bot"], out["aktier"][t]["behall"]
        print(f"   bot {b['arlig']:+.1%}/år (Sharpe {b['sharpe']:.2f})  "
              f"behåll {h['arlig']:+.1%}/år (Sharpe {h['sharpe']:.2f})")
    json.dump(out, open(F_BACKTEST, "w"), ensure_ascii=False, indent=1)
    print(f"Sparat i {os.path.basename(F_BACKTEST)}")


# ---------------- Nyheter ----------------
def fetch_news(ticker: str) -> list[dict]:
    import yfinance as yf
    cutoff = datetime.now(timezone.utc) - timedelta(days=NEWS_DAYS)
    items = []
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    for n in raw:
        c = n.get("content", n)
        title = c.get("title") or ""
        summary = c.get("summary") or ""
        date = c.get("pubDate") or ""
        try:
            dt = datetime.fromisoformat(date.replace("Z", "+00:00"))
        except Exception:
            continue
        if dt >= cutoff and title:
            items.append({"rubrik": title, "sammanfattning": summary,
                          "datum": dt.strftime("%Y-%m-%d")})
    return items


def sentiment_claude(ticker: str, items: list[dict]) -> list[dict] | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or not items:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        listing = "\n".join(f"{i}. {x['rubrik']} — {x['sammanfattning']}"
                            for i, x in enumerate(items))
        prompt = (
            f"Du bedömer nyheter för aktien {ticker}. För varje nyhet: avgör om den "
            f"verkligen handlar om just det bolaget (relevant true/false; t.ex. är "
            f"Volvo Cars INTE samma bolag som AB Volvo VOLV-B.ST) och ge en poäng från "
            f"-1 (mycket negativt för aktiekursen) till 1 (mycket positivt), samt en "
            f"kort motivering på svenska.\n\n{listing}\n\n"
            f'Svara ENDAST med JSON: [{{"nr":0,"relevant":true,"poang":0.3,"motivering":"..."}}]')
        msg = client.messages.create(model=CLAUDE_MODEL, max_tokens=1500,
                                     messages=[{"role": "user", "content": prompt}])
        text = msg.content[0].text
        res = json.loads(text[text.index("["):text.rindex("]") + 1])
        for r in res:
            i = int(r["nr"])
            if 0 <= i < len(items):
                items[i]["poang"] = float(r["poang"]) if r.get("relevant", True) else 0.0
                items[i]["relevant"] = bool(r.get("relevant", True))
                items[i]["motivering"] = r.get("motivering", "")
                items[i]["metod"] = "Claude"
        return items
    except Exception as e:
        print(f"   Claude-analys misslyckades ({e}), använder VADER")
        return None


def sentiment_vader(items: list[dict]) -> list[dict]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    for x in items:
        x["poang"] = sia.polarity_scores(f"{x['rubrik']}. {x['sammanfattning']}")["compound"]
        x["relevant"] = True
        x["motivering"] = ""
        x["metod"] = "VADER"
    return items


def news_score(ticker: str) -> tuple[float, list[dict]]:
    items = fetch_news(ticker)
    if not items:
        return 0.0, []
    items = sentiment_claude(ticker, items) or sentiment_vader(items)
    rel = [x["poang"] for x in items if x.get("relevant", True)]
    return (float(np.mean(rel)) if rel else 0.0), items


# ---------------- Dagens signal ----------------
def signal_today(ticker: str) -> dict:
    df, data, cols = dataset(ticker)
    full = data.dropna(subset=["y"])
    model = HistGradientBoostingClassifier(**MODEL_PARAMS).fit(full[cols], full["y"])
    recent = data[cols].iloc[-SMOOTH:]
    p_ml = float(model.predict_proba(recent)[:, 1].mean())
    # Vad tittar modellen mest på? (testas på senaste årets kända utfall)
    last = full.iloc[-252:]
    imp = permutation_importance(model, last[cols], last["y"], n_repeats=3,
                                 random_state=0, scoring="roc_auc")
    top = sorted(zip(cols, imp.importances_mean), key=lambda z: -z[1])[:5]
    sent, items = news_score(ticker)
    score = float(np.clip(p_ml + NEWS_WEIGHT * sent, 0, 1))
    return {"aktie": ticker, "datum": df.index[-1].strftime("%Y-%m-%d"),
            "pris": float(df["Close"].iloc[-1]), "ml": p_ml, "nyheter": sent,
            "poang": score, "nyhetslista": items,
            "viktigast": [(n, float(v)) for n, v in top]}


# ---------------- Papperstrading ----------------
def load_portfolio() -> dict:
    if os.path.exists(F_PORTFOLIO):
        return json.load(open(F_PORTFOLIO))
    return {"start": datetime.now().strftime("%Y-%m-%d"), "startkapital": START_CASH,
            "kassa": START_CASH, "innehav": {}, "senaste_signaler": []}


def append_csv(path: str, rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def evaluate_predictions():
    """Fyll i facit för gamla prediktioner – så ser vi om boten har rätt i verkligheten."""
    if not os.path.exists(F_PRED):
        return
    pred = pd.read_csv(F_PRED)
    for i, r in pred[pred["utfall"].isna()].iterrows():
        try:
            df = load_data(r["aktie"])
        except Exception:
            continue
        after = df[df.index > pd.Timestamp(r["datum"])]
        if len(after) >= HORIZON:
            ret = float(after["Close"].iloc[HORIZON - 1] / r["pris"] - 1)
            pred.loc[i, "utfall"] = ret
            pred.loc[i, "ratt_ml"] = int((ret > 2 * COST) == (r["ml"] >= 0.5))
            pred.loc[i, "ratt_med_nyheter"] = int((ret > 2 * COST) == (r["poang"] >= 0.5))
    pred.to_csv(F_PRED, index=False)


def cmd_papper():
    pf = load_portfolio()
    fx = usdsek()
    print(f"Papperstrading {datetime.now():%Y-%m-%d}  (USD/SEK {fx:.2f})")
    evaluate_predictions()

    sigs = []
    for t in TICKERS:
        try:
            print(f"   analyserar {t}...")
            sigs.append(signal_today(t))
        except Exception as e:
            print(f"   {t}: hoppar över ({e})")

    prices = {s["aktie"]: to_sek(s["aktie"], s["pris"], fx) for s in sigs}
    holdings = {t: float(n) for t, n in pf["innehav"].items()}
    equity = pf["kassa"] + sum(n * prices.get(t, 0) for t, n in holdings.items())
    slot = equity / len(TICKERS)  # lika stor del per aktie

    trades, today = [], datetime.now().strftime("%Y-%m-%d")
    # Sälj först, köp sen
    for s in sorted(sigs, key=lambda s: s["poang"]):
        t, price = s["aktie"], prices[s["aktie"]]
        have = holdings.get(t, 0.0)
        if s["poang"] < SELL_THRESHOLD and have > 0:
            value = have * price
            pf["kassa"] += value * (1 - COST)
            holdings.pop(t)
            trades.append({"datum": today, "aktie": t, "handling": "SÄLJ",
                           "antal": round(have, 4), "pris_sek": round(price, 2),
                           "belopp_sek": round(value, 2), "poang": round(s["poang"], 3)})
        s["handling"] = "SÄLJ" if s["poang"] < SELL_THRESHOLD else "BEHÅLL"
    for s in sorted(sigs, key=lambda s: -s["poang"]):
        t, price = s["aktie"], prices[s["aktie"]]
        if s["poang"] >= BUY_THRESHOLD:
            s["handling"] = "KÖP/BEHÅLL"
            want = slot - holdings.get(t, 0.0) * price
            spend = min(want, pf["kassa"])
            if spend > slot * 0.2:  # undvik småaffärer
                n = spend * (1 - COST) / price
                holdings[t] = holdings.get(t, 0.0) + n
                pf["kassa"] -= spend
                trades.append({"datum": today, "aktie": t, "handling": "KÖP",
                               "antal": round(n, 4), "pris_sek": round(price, 2),
                               "belopp_sek": round(spend, 2), "poang": round(s["poang"], 3)})

    pf["innehav"] = holdings
    equity = pf["kassa"] + sum(n * prices[t] for t, n in holdings.items())
    # Jämförelse: om man hade köpt SPY med alla pengarna första dagen
    spy_sek = prices.get(MARKET)
    pf.setdefault("spy_start_sek", spy_sek)
    bench = START_CASH * spy_sek / pf["spy_start_sek"] if spy_sek else None

    pf["senaste_signaler"] = [{k: v for k, v in s.items()} for s in sigs]
    pf["uppdaterad"] = datetime.now().isoformat(timespec="minutes")
    json.dump(pf, open(F_PORTFOLIO, "w"), ensure_ascii=False, indent=1)
    append_csv(F_TRADES, trades)
    hist = pd.read_csv(F_HISTORY) if os.path.exists(F_HISTORY) else pd.DataFrame()
    if len(hist) and (hist["datum"] == today).any():
        hist = hist[hist["datum"] != today]  # skriv över om körd två gånger samma dag
    row = {"datum": today, "varde": round(equity, 2), "kassa": round(pf["kassa"], 2),
           "jamforelse_spy": round(bench, 2) if bench else None}
    pd.concat([hist, pd.DataFrame([row])]).to_csv(F_HISTORY, index=False)
    append_csv(F_PRED, [{"datum": s["datum"], "aktie": s["aktie"], "pris": s["pris"],
                         "ml": round(s["ml"], 4), "nyheter": round(s["nyheter"], 4),
                         "poang": round(s["poang"], 4), "utfall": None,
                         "ratt_ml": None, "ratt_med_nyheter": None} for s in sigs])

    print(f"\nAffärer idag: {len(trades)}")
    for tr in trades:
        print(f"   {tr['handling']:5} {tr['aktie']:10} {tr['belopp_sek']:>10,.0f} kr")
    print(f"Portföljvärde: {equity:,.0f} kr (start {START_CASH:,} kr, "
          f"{equity / START_CASH - 1:+.2%})")


# ---------------- Rapport ----------------
def fig_to_img(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def cmd_rapport():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    parts = []

    # 1. Papperstrading
    if os.path.exists(F_HISTORY):
        pf = json.load(open(F_PORTFOLIO))
        h = pd.read_csv(F_HISTORY)
        last = h.iloc[-1]
        diff = last["varde"] / START_CASH - 1
        parts.append(f"<h2>Låtsaspengar</h2><div class='kpi'><b>{last['varde']:,.0f} kr</b>"
                     f"<span class='{'pos' if diff >= 0 else 'neg'}'>{diff:+.2%}</span>"
                     f"<small>sedan {pf['start']} · kassa {last['kassa']:,.0f} kr</small></div>")
        if len(h) > 1:
            fig, ax = plt.subplots(figsize=(7, 2.6))
            ax.plot(pd.to_datetime(h["datum"]), h["varde"], label="Boten", color="#2563eb", lw=2)
            if h["jamforelse_spy"].notna().any():
                ax.plot(pd.to_datetime(h["datum"]), h["jamforelse_spy"],
                        label="Om allt låg i SPY", color="#9ca3af", lw=1.5)
            ax.axhline(START_CASH, color="#e5e7eb", lw=1)
            ax.legend(frameon=False)
            parts.append(fig_to_img(fig))
        sig = pf.get("senaste_signaler", [])
        if sig:
            rows = "".join(
                f"<tr><td>{s['aktie']}</td><td>{s['ml']:.0%}</td><td>{s['nyheter']:+.2f}</td>"
                f"<td><b>{s['poang']:.0%}</b></td><td>{s.get('handling', '')}</td>"
                f"<td>{pf['innehav'].get(s['aktie'], 0):.2f}</td></tr>" for s in sig)
            parts.append("<h2>Dagens beslut</h2><table><tr><th>Aktie</th><th>ML</th>"
                         "<th>Nyheter</th><th>Poäng</th><th>Beslut</th><th>Antal ägda</th></tr>"
                         f"{rows}</table><p class='note'>Köp över {BUY_THRESHOLD:.0%}, sälj under "
                         f"{SELL_THRESHOLD:.0%}. Nyheter flyttar poängen max ±{NEWS_WEIGHT:.0%}.</p>")
            news_html = ""
            for s in sig:
                for n in s.get("nyhetslista", [])[:4]:
                    cls = "pos" if n["poang"] > 0.1 else "neg" if n["poang"] < -0.1 else ""
                    rel = "" if n.get("relevant", True) else " <i>(fel bolag – räknas ej)</i>"
                    why = f"<br><small>{esc(n['motivering'])}</small>" if n.get("motivering") else ""
                    news_html += (f"<li><span class='{cls}'>{n['poang']:+.2f}</span> "
                                  f"<b>{s['aktie']}</b> {esc(n['rubrik'])}{rel}{why}</li>")
            if news_html:
                metod = next((n["metod"] for s in sig for n in s.get("nyhetslista", [])), "")
                parts.append(f"<h2>Nyheter boten läst ({metod})</h2><ul class='news'>{news_html}</ul>")
            imp = "".join(f"<li><b>{s['aktie']}:</b> " +
                          ", ".join(n for n, _ in s.get("viktigast", [])[:3]) + "</li>" for s in sig)
            parts.append(f"<h2>Vad modellen tittar mest på</h2><ul>{imp}</ul>")
    if os.path.exists(F_TRADES):
        tr = pd.read_csv(F_TRADES).tail(15).iloc[::-1]
        rows = "".join(f"<tr><td>{r.datum}</td><td>{r.handling}</td><td>{r.aktie}</td>"
                       f"<td>{r.belopp_sek:,.0f} kr</td></tr>" for r in tr.itertuples())
        parts.append(f"<h2>Senaste affärer</h2><table>{rows}</table>")

    # 2. Facit på riktiga prediktioner
    if os.path.exists(F_PRED):
        p = pd.read_csv(F_PRED).dropna(subset=["utfall"])
        if len(p):
            parts.append(f"<h2>Facit i verkligheten</h2><p>{len(p)} prediktioner har fått facit. "
                         f"ML rätt: <b>{p['ratt_ml'].mean():.0%}</b> · med nyheter: "
                         f"<b>{p['ratt_med_nyheter'].mean():.0%}</b></p>")
        else:
            parts.append(f"<h2>Facit i verkligheten</h2><p class='note'>Första facit kommer "
                         f"efter {HORIZON} handelsdagar (ungefär en månad).</p>")

    # 3. Hur den lärt sig (backtest)
    if os.path.exists(F_BACKTEST):
        bt = json.load(open(F_BACKTEST))
        fig, ax = plt.subplots(figsize=(7, 2.8))
        for t, a in bt["aktier"].items():
            L = pd.DataFrame(a["larande"])
            if len(L):
                ax.plot(pd.to_datetime(L["period"]), L["auc"].rolling(3, min_periods=1).mean(),
                        lw=1.2, label=t)
        ax.axhline(0.5, color="#ef4444", ls="--", lw=1)
        ax.text(ax.get_xlim()[0], 0.505, " 0,5 = slumpen", color="#ef4444", fontsize=8)
        ax.set_ylabel("AUC (högre = bättre)")
        ax.legend(frameon=False, ncol=4, fontsize=7)
        parts.append("<h2>Hur den lärt sig genom åren</h2><p class='note'>Varje punkt är ett halvår "
                     "modellen aldrig sett. Över 0,5 betyder att den gissar bättre än slumpen.</p>"
                     + fig_to_img(fig))
        rows = "".join(
            f"<tr><td>{t}</td><td>{a['bot']['arlig']:+.1%}</td><td>{a['behall']['arlig']:+.1%}</td>"
            f"<td>{a['bot']['max_ras']:.0%}</td><td>{a['behall']['max_ras']:.0%}</td>"
            f"<td>{np.mean([l['auc'] for l in a['larande']]):.2f}</td></tr>"
            for t, a in bt["aktier"].items())
        parts.append("<h2>Backtest 2014–idag</h2><table><tr><th>Aktie</th><th>Bot/år</th>"
                     "<th>Behåll/år</th><th>Bot ras</th><th>Behåll ras</th><th>Snitt-AUC</th></tr>"
                     f"{rows}</table>")

    html = f"""<!DOCTYPE html><html lang="sv"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Tradingbot</title>
<meta http-equiv="refresh" content="1800"><link rel="manifest" href="manifest.json">
<meta name="apple-mobile-web-app-capable" content="yes"><meta name="theme-color" content="#2563eb">
<link rel="apple-touch-icon" href="ikon.png">
<style>body{{font-family:system-ui,sans-serif;max-width:760px;margin:auto;padding:16px;color:#111;background:#fafafa}}
h1{{font-size:22px}}h2{{font-size:16px;margin-top:28px;border-bottom:1px solid #e5e7eb;padding-bottom:4px}}
img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{padding:5px 6px;border-bottom:1px solid #eee;text-align:left}}
.kpi b{{font-size:30px;margin-right:10px}}.kpi small{{display:block;color:#666}}
.pos{{color:#16a34a;font-weight:600}}.neg{{color:#dc2626;font-weight:600}}
.note{{color:#666;font-size:13px}}.news li{{margin-bottom:8px;font-size:13px}}
.warn{{background:#fff7ed;border:1px solid #fed7aa;padding:10px;border-radius:8px;font-size:13px}}</style></head>
<body><h1>Tradingbot</h1><p class="note">Uppdaterad {datetime.now():%Y-%m-%d %H:%M}</p>
<div class="warn">Låtsaspengar. Historiskt resultat är ingen garanti för framtiden.</div>
{''.join(parts)}</body></html>"""
    open(F_REPORT, "w").write(html)
    print(f"Rapport sparad: {os.path.basename(F_REPORT)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "allt"
    if cmd == "backtest":
        cmd_backtest()
    elif cmd == "papper":
        cmd_papper()
    elif cmd == "rapport":
        cmd_rapport()
    elif cmd == "allt":
        cmd_papper()
        cmd_rapport()
    else:
        print(__doc__)
