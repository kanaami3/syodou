"""
初動スコア計算モジュール (Stealth Accumulation Score)
TradingView版「大口買い集め・初動検出」インジケーターのPython移植。

スコア構成 (100点満点):
  ① ステルス蓄積 (OBVダイバージェンス) ... 最大30点
  ② 出来高の質 (静かな増加 + 陽線優位) ... 最大25点
  ③ ボラティリティ収縮 → 放れ          ... 最大20点
  ④ モメンタム転換の初動                ... 最大25点
  + 乗り遅れ防止フィルタ (乖離率/RSI過熱で減点)
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------
# 基本インジケーター
# ----------------------------------------------------------------

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    signal = line.ewm(span=sig, adjust=False).mean()
    return line, signal, line - signal


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def percentrank(s: pd.Series, length: int) -> pd.Series:
    """Pineのta.percentrank相当: 過去length本のうち現在値以下の割合(%)"""
    def _pr(x):
        return (x[:-1] <= x[-1]).mean() * 100

    return s.rolling(length + 1).apply(_pr, raw=True)


def bars_since(cond: pd.Series) -> pd.Series:
    """条件が最後に成立してからのバー数 (未成立ならNaN)"""
    idx = np.arange(len(cond))
    last = np.where(cond.fillna(False).to_numpy(), idx, np.nan)
    last = pd.Series(last, index=cond.index).ffill()
    return pd.Series(idx, index=cond.index) - last


def crossover(a: pd.Series, b) -> pd.Series:
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a > b) & (a.shift(1) <= b.shift(1))


# ----------------------------------------------------------------
# 初動スコア本体
# ----------------------------------------------------------------

DEFAULT_PARAMS = dict(
    div_len=25,        # ①比較期間
    vol_len_s=5,       # ②短期出来高平均
    vol_len_l=25,      # ②長期出来高平均
    ud_len=20,         # ②陽線/陰線出来高の集計期間
    bb_len=20,         # ③BB期間
    sqz_look=100,      # ③収縮判定の参照期間
    rsi_len=14,        # ④RSI期間
    ma_len=25,         # ④基準移動平均
    score_th=65,       # 初動判定スコア
    kairi_cap=6.0,     # 乖離率がこれを超えたら初動とみなさない(%)
    rsi_cap=72.0,      # RSIがこれを超えたら過熱扱い
)


def compute_scores(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    OHLCVのDataFrame (columns: Open, High, Low, Close, Volume) を受け取り、
    スコア内訳・シグナルを列として付加したDataFrameを返す。
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    df = df.copy()
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    # ---------- ① ステルス蓄積 (30点) ----------
    _obv = obv(c, v)
    price_chg = c.pct_change(p["div_len"]) * 100
    obv_range = _obv.rolling(p["sqz_look"]).max() - _obv.rolling(p["sqz_look"]).min()
    obv_chg = (_obv - _obv.shift(p["div_len"])) / obv_range.replace(0, np.nan) * 100

    stealth = np.select(
        [
            (price_chg.abs() < 5) & (obv_chg > 15),
            (price_chg.abs() < 5) & (obv_chg > 8),
            (price_chg > -8) & (price_chg < 3) & (obv_chg > 8),
            obv_chg > 20,
        ],
        [30, 22, 18, 12],
        default=0,
    ).astype(float)

    # OBVが価格より先に高値更新 → ボーナス
    obv_hh = _obv > _obv.rolling(p["div_len"]).max().shift(1)
    price_hh = c > c.rolling(p["div_len"]).max().shift(1)
    stealth = np.minimum(stealth + np.where(obv_hh & ~price_hh, 5, 0), 30)

    # ---------- ② 出来高の質 (25点) ----------
    vol_trend = v.rolling(p["vol_len_s"]).mean() / v.rolling(p["vol_len_l"]).mean()
    up_vol = v.where(c > o, 0.0).rolling(p["ud_len"]).sum()
    dn_vol = v.where(c < o, 0.0).rolling(p["ud_len"]).sum()
    ud_ratio = up_vol / dn_vol.replace(0, np.nan)
    ud_ratio = ud_ratio.fillna(2.0)

    vol_q = (
        np.select([vol_trend > 1.5, vol_trend > 1.2, vol_trend > 1.0], [10, 7, 4], 0)
        + np.select([ud_ratio > 1.6, ud_ratio > 1.3, ud_ratio > 1.1], [15, 11, 6], 0)
    ).astype(float)

    # ---------- ③ 収縮 → 放れ (20点) ----------
    basis = c.rolling(p["bb_len"]).mean()
    dev = c.rolling(p["bb_len"]).std(ddof=0)
    bbw = (dev * 4) / basis.replace(0, np.nan)
    bbw_pct = percentrank(bbw, p["sqz_look"])

    is_sqz = bbw_pct < 25
    was_sqz = bbw_pct.shift(3) < 25
    expanding = bbw > bbw.shift(3)
    break_up = c > h.rolling(15).max().shift(1)

    sqz = np.select(
        [
            was_sqz & expanding & (c > basis),
            is_sqz & (c > basis),
            break_up & (bbw_pct < 50),
        ],
        [20, 12, 14],
        default=0,
    ).astype(float)

    # ---------- ④ モメンタム転換の初動 (25点) ----------
    _rsi = rsi(c, p["rsi_len"])
    ma = c.rolling(p["ma_len"]).mean()
    macd_line, macd_sig, macd_hist = macd(c)

    bs_rsi = bars_since(crossover(_rsi, 50))
    comp_rsi = np.where(
        (bs_rsi <= 5) & (_rsi < 65), 8,
        np.where((_rsi >= 45) & (_rsi < 62), 5, 0),
    )

    gc_below = crossover(macd_line, macd_sig) & (macd_line < 0)
    bs_gc = bars_since(gc_below)
    comp_macd = np.where(
        bs_gc <= 5, 9,
        np.where((macd_line > macd_sig) & (macd_line < 0), 6,
                 np.where(macd_line > macd_sig, 3, 0)),
    )

    bs_ma = bars_since(crossover(c, ma))
    comp_ma = np.where(bs_ma <= 5, 8, np.where(c > ma, 4, 0))

    momentum = (comp_rsi + comp_macd + comp_ma).astype(float)

    # ---------- 合計 & 初動フィルタ ----------
    kairi = (c - ma) / ma.replace(0, np.nan) * 100
    raw_score = stealth + vol_q + sqz + momentum
    too_late = (kairi > p["kairi_cap"]) | (_rsi > p["rsi_cap"])
    score = np.where(too_late, raw_score * 0.4, raw_score)

    is_signal = (score >= p["score_th"]) & ~too_late
    new_signal = is_signal & ~pd.Series(is_signal, index=df.index).shift(1).fillna(False).to_numpy()

    # ---------- 出力 ----------
    df["score"] = score
    df["stealth"] = stealth
    df["vol_quality"] = vol_q
    df["squeeze_break"] = sqz
    df["momentum"] = momentum
    df["kairi"] = kairi
    df["rsi"] = _rsi
    df["obv"] = _obv
    df["ma25"] = ma
    df["bb_upper"] = basis + dev * 2
    df["bb_lower"] = basis - dev * 2
    df["bbw_pct"] = bbw_pct
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_sig
    df["macd_hist"] = macd_hist
    df["vol_ratio"] = v / v.rolling(p["vol_len_l"]).mean()
    df["is_squeeze"] = is_sqz
    df["too_late"] = too_late
    df["is_signal"] = is_signal
    df["new_signal"] = new_signal
    return df


def state_label(row: pd.Series) -> str:
    """最新バーの状態ラベル"""
    if row["too_late"]:
        return "過熱 (初動ではない)"
    if row["is_signal"]:
        return "初動の可能性"
    if row["is_squeeze"] and row["stealth"] >= 18:
        return "蓄積+収縮 (監視推奨)"
    if row["is_squeeze"]:
        return "収縮中 (監視)"
    return "様子見"


def inflow_start(scored: pd.DataFrame, min_score: float = 50, max_gap: int = 3):
    """
    現在まで続いている「資金流入」の開始日を検出する。

    流入の定義: スコアがmin_score以上、または蓄積スコアが18以上。
    max_gap本以内の途切れは同一の流入期間とみなす。

    Returns:
        (開始日timestamp, 経過バー数)。流入中でなければ (None, 0)。
    """
    cond = ((scored["score"] >= min_score) | (scored["stealth"] >= 18)).fillna(False)
    # 直近max_gap+1本以内に流入がなければ「流入中」ではない
    if not cond.tail(max_gap + 1).any():
        return None, 0
    gap = 0
    start_idx = None
    for i in range(len(cond) - 1, -1, -1):
        if cond.iloc[i]:
            start_idx = i
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
    if start_idx is None:
        return None, 0
    return scored.index[start_idx], len(scored) - start_idx
