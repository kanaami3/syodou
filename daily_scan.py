"""自動スキャン (GitHub Actionsから毎日実行)"""

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from scoring import DEFAULT_PARAMS, compute_scores, inflow_start, state_label

JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
JST = timezone(timedelta(hours=9))

UNIVERSE = os.environ.get("UNIVERSE", "プライム")
LIMIT = int(os.environ.get("LIMIT", "300"))
PERIOD = os.environ.get("PERIOD", "2y")


def load_universe() -> pd.DataFrame:
    try:
        df = pd.read_excel(JPX_URL, dtype=str)
        df = df.rename(columns={"コード": "code", "銘柄名": "name", "市場・商品区分": "segment"})
        df = df[df["segment"].str.contains("プライム|スタンダード|グロース", na=False)].copy()
        df["segment"] = (
            df["segment"].str.replace("（内国株式）", "", regex=False)
            .str.replace("(内国株式)", "", regex=False)
            .str.replace("市場", "").str.strip()
        )
        df["code"] = df["code"].str.strip()
        if UNIVERSE != "全市場":
            df = df[df["segment"] == UNIVERSE]
        print(f"JPX一覧取得OK: {UNIVERSE} {len(df)}銘柄 → 先頭{LIMIT}件をスキャン")
        return df.head(LIMIT).reset_index(drop=True)
    except Exception as e:
        print(f"JPX一覧の取得に失敗 ({e})。tickers.csvを使用します")
        t = pd.read_csv("tickers.csv", dtype={"code": str})
        t["code"] = t["code"].str.strip()
        t["segment"] = "カスタム"
        return t


def main():
    tickers = load_universe()
    rows = []
    for i, r in enumerate(tickers.itertuples()):
        symbol = r.code + ".T"
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period=PERIOD, auto_adjust=True)[
                ["Open", "High", "Low", "Close", "Volume"]
            ].dropna()
            if len(hist) < 150:
                continue
            try:
                mcap = tk.fast_info["marketCap"]
            except Exception:
                mcap = None
            scored = compute_scores(hist)
            last = scored.iloc[-1]
            start, days = inflow_start(scored)
            chg = (hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100 if len(hist) > 1 else 0.0
            rows.append(
                dict(
                    コード=r.code,
                    銘柄名=r.name,
                    市場=r.segment,
                    終値=round(float(hist["Close"].iloc[-1]), 1),
                    前日比=round(float(chg), 2),
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
            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{len(tickers)} 完了")
        except Exception as e:
            print(f"  {r.code} {r.name}: 取得失敗 ({e})")
        time.sleep(0.25)

    df = pd.DataFrame(rows)
    df["スキャン日時"] = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/scan_result.csv", index=False)
    print(f"完了: {len(df)}銘柄を保存 (初動シグナル {int(df['初動'].sum())}件)")


if __name__ == "__main__":
    main()
