"""
大口資金流入・初動スクリーナー (自分用ダッシュボード)

起動方法:
    streamlit run app.py
"""

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from scoring import DEFAULT_PARAMS, compute_scores, inflow_start, state_label

st.set_page_config(page_title="初動スクリーナー", page_icon="📈", layout="wide")

# JPX公式の上場銘柄一覧 (市場区分つき・毎月更新)
JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

# 週足用パラメータ (日足の約1/5スケール)
WEEKLY_PARAMS = dict(
    div_len=13, vol_len_s=3, vol_len_l=13, ud_len=10,
    bb_len=20, sqz_look=52, ma_len=13,
)

# ----------------------------------------------------------------
# 銘柄ユニバース
# ----------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner="JPXから上場銘柄一覧を取得中...")
def load_jpx_list() -> pd.DataFrame:
    df = pd.read_excel(JPX_URL, dtype=str)
    df = df.rename(columns={"コード": "code", "銘柄名": "name", "市場・商品区分": "segment"})
    df = df[df["segment"].str.contains("プライム|スタンダード|グロース", na=False)].copy()
    df["segment"] = (
        df["segment"].str.replace("（内国株式）", "", regex=False)
        .str.replace("(内国株式)", "", regex=False)
        .str.replace("市場", "").str.strip()
    )
    df["code"] = df["code"].str.strip()
    df["symbol"] = df["code"] + ".T"
    return df[["code", "name", "segment", "symbol"]].reset_index(drop=True)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_custom_tickers() -> pd.DataFrame:
    t = pd.read_csv("tickers.csv", dtype={"code": str})
    t["code"] = t["code"].str.strip()
    t["symbol"] = t["code"] + ".T"
    t["segment"] = "カスタム"
    return t[["code", "name", "segment", "symbol"]]


# ----------------------------------------------------------------
# サイドバー: 設定
# ----------------------------------------------------------------
st.sidebar.title("⚙️ 設定")

universe = st.sidebar.selectbox(
    "🏛 対象市場",
    ["カスタム (tickers.csv)", "東証プライム", "東証スタンダード", "東証グロース", "東証全市場"],
)
scan_limit = st.sidebar.slider(
    "スキャン上限銘柄数", 50, 1000, 200, 50,
    help="市場全体は銘柄数が多いため上限を設定。コード順に先頭から取得します",
)
min_mcap = st.sidebar.selectbox(
    "💰 最低時価総額 (億円)", [0, 50, 100, 300, 500, 1000, 3000, 10000], index=0,
)

st.sidebar.markdown("---")
score_th = st.sidebar.slider("初動判定スコア", 50, 90, DEFAULT_PARAMS["score_th"])
kairi_cap = st.sidebar.slider("過熱とみなす乖離率 (%)", 3.0, 15.0, DEFAULT_PARAMS["kairi_cap"], 0.5)
period = st.sidebar.selectbox("取得期間", ["2y", "1y", "5y"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("**📊 チャート表示**")
show_macd = st.sidebar.checkbox("MACD", True)
show_rsi = st.sidebar.checkbox("RSI", True)
show_obv = st.sidebar.checkbox("OBV (蓄積)", True)

params = {**DEFAULT_PARAMS, "score_th": score_th, "kairi_cap": kairi_cap}
w_params = {**params, **WEEKLY_PARAMS}


def get_universe() -> pd.DataFrame:
    if universe.startswith("カスタム"):
        return load_custom_tickers()
    try:
        jpx = load_jpx_list()
    except Exception as e:
        st.sidebar.error(f"JPX一覧の取得に失敗しました ({e})。tickers.csvを使用します")
        return load_custom_tickers()
    if universe == "東証プライム":
        jpx = jpx[jpx["segment"] == "プライム"]
    elif universe == "東証スタンダード":
        jpx = jpx[jpx["segment"] == "スタンダード"]
    elif universe == "東証グロース":
        jpx = jpx[jpx["segment"] == "グロース"]
    return jpx.head(scan_limit).reset_index(drop=True)


# ----------------------------------------------------------------
# データ取得 & スキャン
# ----------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_stock(symbol: str, period: str):
    tk = yf.Ticker(symbol)
    hist = tk.history(period=period, auto_adjust=True)[
        ["Open", "High", "Low", "Close", "Volume"]
    ].dropna()
    try:
        mcap = tk.fast_info["marketCap"]
    except Exception:
        mcap = None
    return hist, mcap


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.resample("W-FRI")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )


def scan(tickers: pd.DataFrame, period: str, params: dict):
    rows, details = [], {}
    progress = st.progress(0.0, text="スキャン中...")
    n = len(tickers)
    for i, r in enumerate(tickers.itertuples()):
        progress.progress((i + 1) / n, text=f"スキャン中... {r.code} {r.name} ({i + 1}/{n})")
        try:
            hist, mcap = fetch_stock(r.symbol, period)
            if len(hist) < 150:
                continue
            scored = compute_scores(hist, params)
            last = scored.iloc[-1]
            start, days = inflow_start(scored)
            rows.append(
                dict(
                    コード=r.code,
                    銘柄名=r.name,
                    市場=r.segment,
                    スコア=round(float(last["score"]), 1),
                    流入開始=start.strftime("%m/%d") if start is not None else "-",
                    経過日=days if start is not None else None,
                    時価総額=round(mcap / 1e8) if mcap else None,
                    蓄積=round(float(last["stealth"]), 0),
                    出来高質=round(float(last["vol_quality"]), 0),
                    収縮放れ=round(float(last["squeeze_break"]), 0),
                    転換=round(float(last["momentum"]), 0),
                    乖離率=round(float(last["kairi"]), 1),
                    状態=state_label(last),
                    初動=bool(last["is_signal"]),
                    監視=bool(last["is_squeeze"] and last["stealth"] >= 18),
                )
            )
            details[r.code] = hist
        except Exception as e:
            st.sidebar.warning(f"{r.code} {r.name}: 取得失敗 ({e})")
        time.sleep(0.1)
    progress.empty()
    return pd.DataFrame(rows), details


# ----------------------------------------------------------------
# チャート描画 (日足/週足 共通・マルチペイン)
# ----------------------------------------------------------------
UP = "#e53935"    # 陽線: 赤 (日本式)
DOWN = "#26a69a"  # 陰線: 青緑


def build_chart(d: pd.DataFrame, score_th: int, tf_label: str, is_daily: bool,
                inflow_from=None) -> go.Figure:
    # 表示するペインを組み立て
    panes = [("price", 0.36), ("volume", 0.10)]
    if show_obv:
        panes.append(("obv", 0.12))
    if show_macd:
        panes.append(("macd", 0.13))
    if show_rsi:
        panes.append(("rsi", 0.11))
    panes.append(("score", 0.15))

    total = sum(h for _, h in panes)
    heights = [h / total for _, h in panes]
    titles = {
        "price": f"ローソク足 / MA / ボリンジャーバンド ({tf_label})",
        "volume": "出来高",
        "obv": "OBV (大口の蓄積の痕跡)",
        "macd": "MACD",
        "rsi": "RSI",
        "score": "🔴 資金流入スコア",
    }
    fig = make_subplots(
        rows=len(panes), cols=1, shared_xaxes=True,
        row_heights=heights, vertical_spacing=0.025,
        subplot_titles=[titles[k] for k, _ in panes],
    )
    row_of = {k: i + 1 for i, (k, _) in enumerate(panes)}

    # ---- 価格ペイン ----
    r = row_of["price"]
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="価格",
        increasing_line_color=UP, increasing_fillcolor=UP,
        decreasing_line_color=DOWN, decreasing_fillcolor=DOWN,
    ), row=r, col=1)
    ma_long_len = 75 if is_daily else 26
    ma_long = d["Close"].rolling(ma_long_len).mean()
    fig.add_trace(go.Scatter(x=d.index, y=d["ma25"], name="短期MA",
                             line=dict(color="#ff9800", width=1.3)), row=r, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=ma_long, name="長期MA",
                             line=dict(color="#7e57c2", width=1.3)), row=r, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["bb_upper"], name="BB上",
                             line=dict(color="rgba(120,144,156,0.7)", width=0.7)), row=r, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["bb_lower"], name="BB下",
                             line=dict(color="rgba(120,144,156,0.7)", width=0.7),
                             fill="tonexty", fillcolor="rgba(120,144,156,0.07)"), row=r, col=1)

    sig = d[d["new_signal"]]
    if len(sig):
        fig.add_trace(go.Scatter(
            x=sig.index, y=sig["Low"] * 0.98, mode="markers+text",
            text=["▲初動"] * len(sig), textposition="bottom center",
            textfont=dict(color="#00c853", size=11),
            marker=dict(symbol="triangle-up", size=13, color="#00c853"),
            name="初動シグナル",
        ), row=r, col=1)

    # ---- 出来高ペイン ----
    r = row_of["volume"]
    vol_colors = [UP if c >= o else DOWN for c, o in zip(d["Close"], d["Open"])]
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=vol_colors,
                         marker_line_width=0, opacity=0.7, name="出来高"), row=r, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["Volume"].rolling(25 if is_daily else 13).mean(),
                             name="出来高MA", line=dict(color="#ffb300", width=1)), row=r, col=1)

    # ---- OBVペイン ----
    if show_obv:
        r = row_of["obv"]
        fig.add_trace(go.Scatter(x=d.index, y=d["obv"], name="OBV",
                                 line=dict(color="#42a5f5", width=1.6)), row=r, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["obv"].rolling(20).mean(), name="OBV MA",
                                 line=dict(color="rgba(66,165,245,0.4)", width=1, dash="dot")), row=r, col=1)

    # ---- MACDペイン ----
    if show_macd:
        r = row_of["macd"]
        hist_colors = ["#e53935" if v >= 0 else "#26a69a" for v in d["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=d.index, y=d["macd_hist"], marker_color=hist_colors,
                             marker_line_width=0, opacity=0.55, name="ヒストグラム"), row=r, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["macd_line"], name="MACD",
                                 line=dict(color="#29b6f6", width=1.3)), row=r, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["macd_signal"], name="シグナル",
                                 line=dict(color="#ff7043", width=1.3)), row=r, col=1)
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.25)", line_width=0.7, row=r, col=1)

    # ---- RSIペイン ----
    if show_rsi:
        r = row_of["rsi"]
        fig.add_trace(go.Scatter(x=d.index, y=d["rsi"], name="RSI",
                                 line=dict(color="#ab47bc", width=1.4)), row=r, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(229,57,53,0.5)", row=r, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(160,160,160,0.4)", row=r, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(38,166,154,0.5)", row=r, col=1)
        fig.update_yaxes(range=[0, 100], row=r, col=1)

    # ---- スコアペイン ----
    r = row_of["score"]
    fig.add_trace(go.Scatter(
        x=d.index, y=d["score"], name="資金流入スコア",
        line=dict(color="#ff1744", width=2),
        fill="tozeroy", fillcolor="rgba(255,23,68,0.08)",
    ), row=r, col=1)
    fig.add_hline(y=score_th, line_dash="dash", line_color="#00c853", row=r, col=1)
    fig.update_yaxes(range=[0, 100], row=r, col=1)

    # ---- 資金流入期間のシェーディング (開始日〜現在) ----
    if inflow_from is not None and inflow_from >= d.index[0]:
        fig.add_vrect(
            x0=inflow_from, x1=d.index[-1],
            fillcolor="rgba(255,23,68,0.06)", line_width=0,
        )
        fig.add_vline(
            x=inflow_from, line_dash="dash", line_color="#ff1744", line_width=1.2,
        )
        fig.add_annotation(
            x=inflow_from, y=1, yref="paper", yanchor="bottom",
            text=f"🔴 資金流入開始 {inflow_from.strftime('%m/%d')}",
            showarrow=False, font=dict(color="#ff1744", size=12),
            bgcolor="rgba(0,0,0,0.4)",
        )

    # ---- 全体の見やすさ調整 ----
    fig.update_layout(
        height=200 + 170 * len(panes),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        margin=dict(t=40, b=20, l=10, r=10),
        hovermode="x unified",
        hoverlabel=dict(font_size=11),
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=0.5,
                     spikedash="dot", spikecolor="rgba(160,160,160,0.6)")
    if is_daily:
        # 土日の隙間を詰める
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=12)
        ann["x"] = 0
        ann["xanchor"] = "left"
    return fig


# ----------------------------------------------------------------
# 数値パネル (チャート下)
# ----------------------------------------------------------------
def render_metrics(last: pd.Series, score_th: int, inflow_from=None, inflow_bars: int = 0,
                   unit: str = "営業日"):
    score = float(last["score"])
    vol_ratio = float(last["vol_ratio"]) if pd.notna(last["vol_ratio"]) else 0
    if inflow_from is not None:
        inflow_txt = f"{inflow_from.strftime('%m/%d')}<span style='font-size:13px'> ({inflow_bars}{unit}経過)</span>"
    else:
        inflow_txt = "-"
    items = [
        ("🔴 資金流入スコア", f"{score:.1f} <span style='font-size:14px'>/100</span>",
         "#ff1744", 30),
        ("🔴 流入開始日", inflow_txt, "#ff1744" if inflow_from is not None else None, 24),
        ("蓄積 (OBV先行)", f"{last['stealth']:.0f}/30", None, 22),
        ("出来高倍率", f"{vol_ratio:.2f}x", "#ff1744" if vol_ratio >= 1.5 else None, 22),
        ("MACDヒスト", f"{last['macd_hist']:.1f}",
         "#e53935" if last["macd_hist"] >= 0 else "#26a69a", 22),
        ("RSI", f"{last['rsi']:.0f}", "#e53935" if last["rsi"] >= 70 else None, 22),
        ("乖離率", f"{last['kairi']:+.1f}%",
         "#e53935" if abs(last["kairi"]) >= 6 else None, 22),
    ]
    cells = ""
    for label, value, color, size in items:
        c = color or "inherit"
        weight = "700" if color else "500"
        cells += (
            f"<div style='flex:1; min-width:120px; text-align:center; padding:8px 4px;'>"
            f"<div style='font-size:12px; opacity:0.65;'>{label}</div>"
            f"<div style='font-size:{size}px; font-weight:{weight}; color:{c};'>{value}</div>"
            f"</div>"
        )
    judge = "🟢 初動の可能性" if last["is_signal"] else ("🔥 過熱" if last["too_late"] else "⚪ 様子見")
    st.markdown(
        f"<div style='display:flex; flex-wrap:wrap; align-items:center; gap:4px; "
        f"border:1px solid rgba(128,128,128,0.3); border-radius:10px; padding:6px;'>"
        f"{cells}"
        f"<div style='flex:1; min-width:120px; text-align:center; font-size:15px; font-weight:600;'>{judge}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------
# メイン画面
# ----------------------------------------------------------------
st.title("📈 大口資金流入・初動スクリーナー")
st.caption(
    "価格がまだ動いていないのに出来高・資金フローが先行している銘柄 (=大口の買い集めの痕跡) を検出します。"
    " データ: Yahoo Finance 日足 / 投資判断は自己責任で。"
)

if st.button(f"🔍 スキャン実行 ({universe})", type="primary"):
    tickers = get_universe()
    result, details = scan(tickers, period, params)
    st.session_state["result"] = result
    st.session_state["details"] = details

if "result" not in st.session_state:
    st.info("対象市場を選んで「スキャン実行」を押してください (銘柄数によっては数分かかります)")
    st.stop()

result: pd.DataFrame = st.session_state["result"]
details: dict = st.session_state["details"]

# ---------- サマリー ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("スキャン銘柄数", len(result))
c2.metric("🟢 初動シグナル", int(result["初動"].sum()))
c3.metric("🔵 蓄積+収縮 (監視)", int(result["監視"].sum()))
c4.metric("平均スコア", round(result["スコア"].mean(), 1))

st.markdown("---")

# ================================================================
# 左: 銘柄リスト / 右: チャート
# ================================================================
col_list, col_chart = st.columns([0.38, 0.62], gap="medium")

# ---------- 左カラム: 銘柄リスト ----------
with col_list:
    st.markdown("#### 📋 銘柄リスト (クリックでチャート表示)")

    search = st.text_input(
        "🔎 銘柄検索", placeholder="コードまたは銘柄名で検索 (例: 7203, トヨタ)",
        label_visibility="collapsed",
    )
    view = st.radio(
        "表示", ["すべて", "初動シグナルのみ", "監視候補のみ"],
        horizontal=True, label_visibility="collapsed",
    )

    shown = result.copy()
    if min_mcap > 0:
        shown = shown[shown["時価総額"].fillna(0) >= min_mcap]
    if search.strip():
        q = search.strip()
        shown = shown[
            shown["コード"].str.contains(q, case=False, na=False)
            | shown["銘柄名"].str.contains(q, case=False, na=False)
        ]
    if view == "初動シグナルのみ":
        shown = shown[shown["初動"]]
    elif view == "監視候補のみ":
        shown = shown[shown["監視"]]

    shown = shown.sort_values("スコア", ascending=False).reset_index(drop=True)

    if len(shown) == 0:
        st.warning("該当銘柄がありません (検索条件・フィルタを見直してください)")
        st.stop()

    def _row_style(row):
        if row["初動"]:
            return ["background-color: rgba(0, 200, 80, 0.15)"] * len(row)
        if row["監視"]:
            return ["background-color: rgba(0, 120, 255, 0.12)"] * len(row)
        return [""] * len(row)

    list_cols = ["コード", "銘柄名", "スコア", "流入開始", "経過日", "時価総額", "状態"]
    event = st.dataframe(
        shown[list_cols].style.apply(
            lambda r: _row_style(shown.loc[r.name]), axis=1
        ).format({"時価総額": lambda v: f"{v:,.0f}億" if pd.notna(v) else "-"}),
        use_container_width=True,
        height=520,
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
        key="stock_table",
    )

    sel_rows = event.selection.rows if event and event.selection else []
    sel_idx = sel_rows[0] if sel_rows else 0
    sel = shown.iloc[sel_idx]

    with st.expander("スコア内訳", expanded=True):
        b1, b2 = st.columns(2)
        b1.metric("① 蓄積", f"{sel['蓄積']:.0f}/30")
        b2.metric("② 出来高質", f"{sel['出来高質']:.0f}/25")
        b3, b4 = st.columns(2)
        b3.metric("③ 収縮放れ", f"{sel['収縮放れ']:.0f}/20")
        b4.metric("④ 転換", f"{sel['転換']:.0f}/25")

# ---------- 右カラム: チャート (日足/週足タブ) ----------
with col_chart:
    mcap_txt = f"{sel['時価総額']:,.0f}億円" if pd.notna(sel["時価総額"]) else "-"
    st.markdown(
        f"#### 📊 {sel['コード']} {sel['銘柄名']}　"
        f"[{sel['市場']}] 時価総額 {mcap_txt}"
    )

    hist = details[sel["コード"]]
    tab_d, tab_w = st.tabs(["📅 日足", "🗓 週足"])

    with tab_d:
        d = compute_scores(hist, params).tail(200)
        d_start, d_bars = inflow_start(d)
        render_metrics(d.iloc[-1], score_th, d_start, d_bars, unit="営業日")
        st.plotly_chart(build_chart(d, score_th, "日足", is_daily=True, inflow_from=d_start),
                        use_container_width=True, key="chart_daily")

    with tab_w:
        wk = to_weekly(hist)
        if len(wk) < 60:
            st.warning("週足の計算にはデータが不足しています。サイドバーの取得期間を2y以上にして再スキャンしてください")
        else:
            w = compute_scores(wk, w_params).tail(150)
            w_start, w_bars = inflow_start(w)
            render_metrics(w.iloc[-1], score_th, w_start, w_bars, unit="週")
            st.plotly_chart(build_chart(w, score_th, "週足", is_daily=False, inflow_from=w_start),
                            use_container_width=True, key="chart_weekly")
            st.caption("週足はパラメータを週足スケールに調整して計算 (MA=13週, 収縮判定=52週)")

st.caption(
    "見方: 価格が横ばいなのにOBVが右肩上がり = 水面下の買い集め。"
    "日足で初動シグナル + 週足でも蓄積が確認できれば信頼度が上がります。"
)
