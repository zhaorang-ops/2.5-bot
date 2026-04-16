import os
import time
import json
import math
import random
from typing import Optional, List, Tuple

import requests
import numpy as np
from datetime import datetime, timezone, timedelta

# ======================================
# 基础配置
# ======================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
]

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "TONUSDT",
    "BNBUSDT",
    "LTCUSDT",
    "SSVUSDT",
    "DOGEUSDT",
]

INTERVALS = ["1h", "4h"]

PRIORITY = {
    "BTCUSDT": 1,
    "ETHUSDT": 2,
    "BNBUSDT": 3,
    "TONUSDT": 4,
    "LTCUSDT": 5,
    "SSVUSDT": 6,
    "DOGEUSDT": 7,
}

PRIORITY_BONUS = {
    "BTCUSDT": 3,
    "ETHUSDT": 2,
    "BNBUSDT": 1,
    "TONUSDT": 0,
    "LTCUSDT": 0,
    "SSVUSDT": 0,
    "DOGEUSDT": 0,
}

# 中国/新加坡同属 UTC+8
UTC8 = timezone(timedelta(hours=8))

# ======================================
# 请求控制
# ======================================
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 20
MAX_RETRIES_PER_URL = 3
RETRY_SLEEP_SECONDS = 1.2

TELEGRAM_CONNECT_TIMEOUT = 5
TELEGRAM_READ_TIMEOUT = 20
TELEGRAM_MAX_RETRIES = 3
TELEGRAM_RETRY_SLEEP_SECONDS = 1.5

# ======================================
# 通知策略
# ======================================
DEDUPE_LEVELS = {
    "no_long": True,
    "watch": True,
    "warning": True,
    "entry": True,
}

ENTRY_BURST_COUNT = 1
ENTRY_BURST_GAP_SECONDS = 2
PIN_ENTRY_MESSAGE = True
UNPIN_PREVIOUS_ENTRY_BEFORE_PIN = True
PIN_SILENT = True

# ======================================
# 策略阈值
# ======================================
STRATEGY_CONFIG = {
    "ENTRY_MIN_SCORE": 75,
    "WARNING_MIN_SCORE": 60,
    "WATCH_MIN_SCORE": 40,
    "ENTRY_MIN_CONFIRM_SCORE": 14,
    "ENTRY_MIN_RR": 1.8,
    "WARNING_MIN_RR": 1.4,
}

BACKTEST_STATS_JSON_FILE = "backtest_overall_stats.json"

# ======================================
# 运行时状态
# ======================================
LAST_SENT_SIGNATURES = {}
LAST_PINNED_MESSAGE_ID = None

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (github-actions-signal-bot)"
})

# ======================================
# 工具函数
# ======================================
def now_cn():
    return datetime.now(UTC8)

def fmt_time_full(dt):
    return dt.strftime("%Y-%m-%d %H:%M")

def fmt_price(symbol: str, price: float) -> str:
    if symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT", "LTCUSDT", "SSVUSDT"):
        return f"{price:.4f}".rstrip("0").rstrip(".")
    return f"{price:.6f}".rstrip("0").rstrip(".")

def clamp(x, low, high):
    return max(low, min(high, x))

def calc_pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return abs(a - b) / abs(b) * 100.0

def short_symbol(symbol: str) -> str:
    return symbol.replace("USDT", "")

def level_priority(level: str) -> int:
    mp = {
        "no_long": 0,
        "watch": 1,
        "warning": 2,
        "entry": 3,
    }
    return mp.get(level, 0)

def interval_rank(interval: str) -> int:
    mp = {"1h": 1, "4h": 2}
    return mp.get(interval, 99)

def rr_to_score(rr: float) -> int:
    if rr >= 2.5:
        return 20
    if rr >= 2.0:
        return 16
    if rr >= 1.8:
        return 14
    if rr >= 1.5:
        return 10
    if rr >= 1.2:
        return 6
    return 0

def suggest_position(score: int, interval: str) -> str:
    if interval == "4h":
        if score >= 85:
            return "30%"
        if score >= 78:
            return "20%"
        return "10%"

    if score >= 84:
        return "20%"
    if score >= 75:
        return "10%"
    return "5%"

def build_dedupe_signature(level: str, interval: str, text: str) -> str:
    lines = []
    for line in text.splitlines():
        clean = line.strip()
        if clean.startswith("时间（中国）："):
            continue
        lines.append(clean)
    normalized = "\n".join(lines)
    return f"{level}:{interval}:{normalized}"

def should_send_message(item: dict) -> bool:
    level = item["level"]
    interval = item["interval"]

    if not DEDUPE_LEVELS.get(level, False):
        item["signature"] = None
        return True

    signature = build_dedupe_signature(level, interval, item["text"])
    item["signature"] = signature
    last_signature = LAST_SENT_SIGNATURES.get(f"{level}:{interval}")
    return signature != last_signature

def remember_sent_message(item: dict):
    signature = item.get("signature")
    if signature:
        LAST_SENT_SIGNATURES[f"{item['level']}:{item['interval']}"] = signature

def load_overall_backtest_stats():
    try:
        with open(BACKTEST_STATS_JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def get_overall_true_win_rate_text() -> str:
    stats = load_overall_backtest_stats()
    if not stats:
        return "暂无历史回测数据"
    return f"{stats.get('win_rate', 0):.2f}%"
    
def estimate_model_win_rate(symbol: str, interval: str, scenario_key: str,
                            score: int, rr: float, confirm_score: int, env_score: int) -> int:
    win = 42

    if score >= 85:
        win += 22
    elif score >= 78:
        win += 18
    elif score >= 72:
        win += 14
    elif score >= 65:
        win += 10
    elif score >= 58:
        win += 6
    else:
        win += 2

    if rr >= 2.5:
        win += 8
    elif rr >= 2.0:
        win += 6
    elif rr >= 1.8:
        win += 5
    elif rr >= 1.5:
        win += 3

    if confirm_score >= 20:
        win += 8
    elif confirm_score >= 16:
        win += 6
    elif confirm_score >= 12:
        win += 4

    if env_score >= 12:
        win += 6
    elif env_score >= 9:
        win += 4
    elif env_score >= 6:
        win += 2

    if symbol == "BTCUSDT":
        win += 4
    elif symbol == "ETHUSDT":
        win += 3
    elif symbol == "BNBUSDT":
        win += 2

    if interval == "4h":
        win += 2

    if scenario_key == "trend_pullback":
        win += 3
    elif scenario_key == "range_bottom":
        win += 1
    elif scenario_key == "oversold_bounce":
        win -= 1

    return int(clamp(win, 35, 88))
# ======================================
# 电脑版表格输出
# ======================================
# ======================================
# 轻量技术指标实现（不依赖 TA-Lib）
# ======================================
def sma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return out
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    out[period - 1:] = (csum[period:] - csum[:-period]) / period
    return out

def ema(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out

def rsi(arr: np.ndarray, period: int = 14) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    if len(arr) <= period:
        return out

    delta = np.diff(arr)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.empty_like(arr, dtype=float)
    avg_loss = np.empty_like(arr, dtype=float)
    avg_gain[:] = np.nan
    avg_loss[:] = np.nan

    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])

    for i in range(period + 1, len(arr)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

    rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
    out[period:] = 100 - (100 / (1 + rs[period:]))
    out[np.isnan(out)] = 50.0
    return out

def macd(arr: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(arr, fast)
    ema_slow = ema(arr, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(np.where(np.isnan(macd_line), 0.0, macd_line), signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bbands(arr: np.ndarray, period: int = 20, nbdev: float = 2.0):
    mid = sma(arr, period)
    out_std = np.full_like(arr, np.nan, dtype=float)
    if len(arr) >= period:
        for i in range(period - 1, len(arr)):
            out_std[i] = np.std(arr[i - period + 1:i + 1], ddof=0)
    upper = mid + nbdev * out_std
    lower = mid - nbdev * out_std
    return upper, mid, lower

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
    tr = np.full_like(close, np.nan, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    out = np.full_like(close, np.nan, dtype=float)
    if len(close) <= period:
        return out
    out[period - 1] = np.nanmean(tr[:period])
    for i in range(period, len(close)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out

# ======================================
# 电脑版表格输出
# ======================================
def truncate_text(s: str, width: int) -> str:
    s = str(s)
    if len(s) <= width:
        return s
    if width <= 1:
        return s[:width]
    return s[:width - 1] + "…"

def pad_cell(s: str, width: int) -> str:
    s = truncate_text(s, width)
    return s.ljust(width)

def build_ascii_table(rows: List[dict], columns: List[Tuple[str, int, str]]) -> str:
    def line_sep():
        return "+" + "+".join("-" * (w + 2) for _, w, _ in columns) + "+"

    header = "| " + " | ".join(pad_cell(title, width) for title, width, _ in columns) + " |"
    lines = [line_sep(), header, line_sep()]

    for row in rows:
        line = "| " + " | ".join(
            pad_cell(row.get(key, ""), width) for _, width, key in columns
        ) + " |"
        lines.append(line)

    lines.append(line_sep())
    return "\n".join(lines)

def print_live_desktop_report(all_rows: List[dict], dt_cn: datetime):
    columns = [
        ("币种", 8, "symbol"),
        ("周期", 4, "interval"),
        ("状态", 8, "status"),
        ("思路", 10, "scenario_name"),
        ("当前价", 12, "latest_price"),
        ("预警区间", 23, "watch_zone"),
        ("做多区间", 23, "entry_zone"),
        ("止损", 12, "stop_loss"),
        ("止盈", 12, "take_profit"),
        ("盈亏比", 7, "rr"),
        ("模型胜率", 8, "model_win_rate"),
        ("总体真胜", 10, "overall_true_win_rate"),
        ("评分", 6, "score"),
    ]

    print("\n" + "=" * 175)
    print(f"桌面行情总表｜时间（中国）：{fmt_time_full(dt_cn)}")
    print("=" * 175)
    print(build_ascii_table(all_rows, columns))
    print("=" * 175 + "\n")

# ======================================
# Telegram
# ======================================
def telegram_api_post(method: str, payload: dict):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return None

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    last_error = None

    for attempt in range(1, TELEGRAM_MAX_RETRIES + 1):
        try:
            r = requests.post(
                url,
                data=payload,
                timeout=(TELEGRAM_CONNECT_TIMEOUT, TELEGRAM_READ_TIMEOUT),
            )
            r.raise_for_status()
            data = r.json()
            if not data.get("ok", False):
                raise RuntimeError(f"Telegram API 错误: {data}")
            return data

        except requests.exceptions.Timeout as e:
            last_error = e
            print(f"[Telegram超时] method={method} attempt={attempt}/{TELEGRAM_MAX_RETRIES} error={e}")

        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"[Telegram请求异常] method={method} attempt={attempt}/{TELEGRAM_MAX_RETRIES} error={e}")

        except Exception as e:
            last_error = e
            print(f"[Telegram其他异常] method={method} attempt={attempt}/{TELEGRAM_MAX_RETRIES} error={e}")

        if attempt < TELEGRAM_MAX_RETRIES:
            sleep_s = TELEGRAM_RETRY_SLEEP_SECONDS * attempt + random.uniform(0, 0.5)
            time.sleep(sleep_s)

    raise RuntimeError(f"Telegram 请求失败（{method}）：{last_error}")

def send_telegram_message(text: str, silent: bool = False):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("未填写 Telegram Token 或 Chat ID，只打印，不发送。")
        print(text)
        print("=" * 100)
        return None

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_notification": silent,
    }
    return telegram_api_post("sendMessage", payload)

def pin_telegram_message(message_id: int, silent: bool = True):
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "message_id": message_id,
        "disable_notification": silent,
    }
    return telegram_api_post("pinChatMessage", payload)

def unpin_telegram_message(message_id: Optional[int] = None):
    payload = {"chat_id": TELEGRAM_CHAT_ID}
    if message_id is not None:
        payload["message_id"] = message_id
    return telegram_api_post("unpinChatMessage", payload)

# ======================================
# Binance 请求
# ======================================
def request_json_with_retry(path: str, params: dict):
    last_error = None

    for base_url in BASE_URLS:
        url = f"{base_url}{path}"

        for attempt in range(1, MAX_RETRIES_PER_URL + 1):
            try:
                r = SESSION.get(
                    url,
                    params=params,
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                )
                r.raise_for_status()
                return r.json()

            except requests.exceptions.Timeout as e:
                last_error = e
                print(f"[超时] {url} params={params} attempt={attempt}/{MAX_RETRIES_PER_URL} error={e}")

            except requests.exceptions.RequestException as e:
                last_error = e
                print(f"[请求异常] {url} params={params} attempt={attempt}/{MAX_RETRIES_PER_URL} error={e}")

            if attempt < MAX_RETRIES_PER_URL:
                sleep_s = RETRY_SLEEP_SECONDS * attempt + random.uniform(0, 0.4)
                time.sleep(sleep_s)

        print(f"[切换备用域名] 当前域名失败：{base_url}")

    raise RuntimeError(f"所有 Binance 接口均请求失败：{last_error}")

def fetch_latest_price(symbol: str) -> float:
    data = request_json_with_retry("/api/v3/ticker/price", {"symbol": symbol})
    return float(data["price"])

def fetch_klines_live(symbol: str, interval: str, limit: int = 300):
    data = request_json_with_retry(
        "/api/v3/klines",
        {"symbol": symbol, "interval": interval, "limit": limit}
    )

    opens = np.array([float(x[1]) for x in data], dtype=float)
    highs = np.array([float(x[2]) for x in data], dtype=float)
    lows = np.array([float(x[3]) for x in data], dtype=float)
    closes = np.array([float(x[4]) for x in data], dtype=float)

    return {
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
    }

# ======================================
# 指标计算
# ======================================
def calc_indicators_live(symbol: str, interval: str):
    data = fetch_klines_live(symbol, interval, 300)

    opens = data["opens"][:-1]
    highs = data["highs"][:-1]
    lows = data["lows"][:-1]
    closes = data["closes"][:-1]

    if len(closes) < 80:
        raise ValueError(f"{symbol} {interval} K线不足")

    rsi_series = rsi(closes, 14)
    ema20_series = ema(closes, 20)
    ema50_series = ema(closes, 50)
    sma20_series = sma(closes, 20)
    sma50_series = sma(closes, 50)
    atr_series = atr(highs, lows, closes, 14)
    macd_line, macd_signal, macd_hist = macd(closes, 12, 26, 9)
    bb_upper, bb_middle, bb_lower = bbands(closes, 20, 2.0)

    last_open = float(opens[-1])
    last_close = float(closes[-1])
    prev_close = float(closes[-2])
    last_high = float(highs[-1])
    last_low = float(lows[-1])

    body = abs(last_close - last_open)
    lower_wick = min(last_open, last_close) - last_low
    upper_wick = last_high - max(last_open, last_close)

    return {
        "symbol": symbol,
        "interval": interval,
        "last_open": last_open,
        "last_close": last_close,
        "prev_close": prev_close,
        "last_high": last_high,
        "last_low": last_low,
        "rsi14": float(rsi_series[-1]),
        "rsi14_prev": float(rsi_series[-2]),
        "ema20": float(ema20_series[-1]),
        "ema50": float(ema50_series[-1]),
        "sma20": float(sma20_series[-1]),
        "sma50": float(sma50_series[-1]),
        "atr14": float(atr_series[-1]),
        "macd": float(macd_line[-1]),
        "macd_signal": float(macd_signal[-1]),
        "macd_hist": float(macd_hist[-1]),
        "macd_hist_prev": float(macd_hist[-2]),
        "bb_upper": float(bb_upper[-1]),
        "bb_middle": float(bb_middle[-1]),
        "bb_lower": float(bb_lower[-1]),
        "recent_low_10": float(np.min(lows[-10:])),
        "recent_high_10": float(np.max(highs[-10:])),
        "recent_low_20": float(np.min(lows[-20:])),
        "recent_high_20": float(np.max(highs[-20:])),
        "recent_low_30": float(np.min(lows[-30:])),
        "recent_high_30": float(np.max(highs[-30:])),
        "breakout_trigger": float(np.max(highs[-21:-1])) if len(highs) >= 21 else float(np.max(highs[:-1])),
        "body": body,
        "lower_wick": lower_wick,
        "upper_wick": upper_wick,
        "lower_wick_ratio": lower_wick / max(body, 1e-9),
        "upper_wick_ratio": upper_wick / max(body, 1e-9),
    }

# ======================================
# 打分系统
# ======================================
def evaluate_environment(ind: dict) -> int:
    score = 0

    if ind["ema20"] > ind["sma50"]:
        score += 4
    elif calc_pct(ind["ema20"], ind["sma50"]) <= 0.8:
        score += 2

    if ind["macd"] > ind["macd_signal"]:
        score += 4
    elif ind["macd_hist"] > ind["macd_hist_prev"]:
        score += 2

    if ind["last_close"] > ind["ema20"]:
        score += 3
    elif ind["last_close"] > ind["bb_middle"]:
        score += 1

    if 35 <= ind["rsi14"] <= 68:
        score += 4
    elif 28 <= ind["rsi14"] < 35:
        score += 2
    elif 68 < ind["rsi14"] <= 72:
        score += 1

    return int(clamp(score, 0, 15))

def compute_confirmation_score(ind: dict, latest_price: float) -> Tuple[int, List[str]]:
    score = 0
    notes = []

    if ind["last_close"] > ind["last_open"]:
        score += 6
        notes.append("上一根K线收阳")

    if ind["lower_wick_ratio"] >= 1.2:
        score += 5
        notes.append("下影线较明显")

    if latest_price > ind["last_close"]:
        score += 6
        notes.append("当前价强于上一收盘")

    if ind["rsi14"] > ind["rsi14_prev"]:
        score += 5
        notes.append("RSI开始回升")

    if ind["macd_hist"] > ind["macd_hist_prev"]:
        score += 6
        notes.append("MACD空头动能减弱")

    if ind["last_low"] >= ind["recent_low_10"] * 0.998:
        score += 2
        notes.append("短线低点未明显破坏")

    return int(clamp(score, 0, 30)), notes

def score_by_location(latest_price: float, zone_low: float, zone_high: float,
                      watch_low: float, watch_high: float, stop: float) -> int:
    if zone_low <= latest_price <= zone_high:
        return 35
    if watch_low <= latest_price <= watch_high:
        return 25
    if latest_price < zone_low and latest_price > stop:
        return 18

    mid = (zone_low + zone_high) / 2.0
    dist_pct = calc_pct(latest_price, mid)

    if dist_pct <= 0.5:
        return 18
    if dist_pct <= 1.0:
        return 12
    if dist_pct <= 2.0:
        return 7
    return 3

def calc_rr(entry: float, stop: float, target: float) -> float:
    risk = entry - stop
    reward = target - entry
    if risk <= 0 or reward <= 0:
        return 0.0
    return reward / risk

def apply_plan_grade(plan: dict, latest_price: float, cfg: dict) -> dict:
    in_entry = plan["entry_zone_low"] <= latest_price <= plan["entry_zone_high"]
    in_watch = plan["watch_zone_low"] <= latest_price <= plan["watch_zone_high"]

    if (
        plan["rr"] >= cfg["ENTRY_MIN_RR"]
        and plan["score"] >= cfg["ENTRY_MIN_SCORE"]
        and in_entry
        and plan["confirm_score"] >= cfg["ENTRY_MIN_CONFIRM_SCORE"]
    ):
        plan["level"] = "entry"
        plan["status"] = "可开多"
        plan["action"] = "已进入做多区间，且盈亏比合适，可执行开多。"
    elif (
        plan["rr"] >= cfg["WARNING_MIN_RR"]
        and plan["score"] >= cfg["WARNING_MIN_SCORE"]
        and (in_watch or in_entry)
    ):
        plan["level"] = "warning"
        plan["status"] = "预警"
        plan["action"] = "已接近或进入做多区间，开始盯盘，等进一步确认。"
    elif plan["score"] >= cfg["WATCH_MIN_SCORE"]:
        plan["level"] = "watch"
        plan["status"] = "观望"
        plan["action"] = "位置开始有价值，但还没到最优开单状态，继续观望。"
    else:
        plan["level"] = "no_long"
        plan["status"] = "暂不做多"
        plan["action"] = "当前价格、确认或盈亏比不理想，暂不做多。"

    plan["model_win_rate"] = estimate_model_win_rate(
        symbol=plan["symbol"],
        interval=plan["interval"],
        scenario_key=plan["scenario_key"],
        score=plan["score"],
        rr=plan["rr"],
        confirm_score=plan["confirm_score"],
        env_score=plan["env_score"],
    )
    return plan

def finalize_plan(symbol: str, interval: str, latest_price: float, ind: dict,
                  env_score: int, scenario_key: str, scenario_name: str,
                  watch_zone_low: float, watch_zone_high: float,
                  entry_zone_low: float, entry_zone_high: float,
                  stop_loss: float, take_profit: float,
                  cfg: dict) -> dict:
    watch_zone_low, watch_zone_high = sorted([watch_zone_low, watch_zone_high])
    entry_zone_low, entry_zone_high = sorted([entry_zone_low, entry_zone_high])

    location_score = score_by_location(
        latest_price,
        entry_zone_low,
        entry_zone_high,
        watch_zone_low,
        watch_zone_high,
        stop_loss,
    )

    confirm_score, confirm_notes = compute_confirmation_score(ind, latest_price)

    reference_entry = latest_price if entry_zone_low <= latest_price <= entry_zone_high else (entry_zone_low + entry_zone_high) / 2.0
    rr = calc_rr(reference_entry, stop_loss, take_profit)
    rr_score = rr_to_score(rr)

    total_score = int(clamp(
        location_score + confirm_score + rr_score + env_score + PRIORITY_BONUS.get(symbol, 0),
        0, 100
    ))

    plan = {
        "symbol": symbol,
        "interval": interval,
        "scenario_key": scenario_key,
        "scenario_name": scenario_name,
        "latest_price": latest_price,
        "env_score": env_score,
        "location_score": location_score,
        "confirm_score": confirm_score,
        "rr_score": rr_score,
        "score": total_score,
        "rr": rr,
        "watch_zone_low": watch_zone_low,
        "watch_zone_high": watch_zone_high,
        "entry_zone_low": entry_zone_low,
        "entry_zone_high": entry_zone_high,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "confirm_notes": confirm_notes,
        "level": "no_long",
        "status": "暂不做多",
        "action": "当前价格、确认或盈亏比不理想，暂不做多。",
        "model_win_rate": 0,
    }

    return apply_plan_grade(plan, latest_price, cfg)

# ======================================
# 三类做多方案
# ======================================
def build_trend_pullback_plan(symbol: str, interval: str, latest_price: float, ind: dict, env_score: int, cfg: dict):
    support = max(ind["ema20"], ind["bb_middle"])
    atr_v = max(ind["atr14"], latest_price * 0.002)

    watch_zone_low = support + 0.15 * atr_v
    watch_zone_high = support + 0.75 * atr_v

    entry_zone_low = support - 0.18 * atr_v
    entry_zone_high = support + 0.12 * atr_v

    stop_loss = min(ind["recent_low_10"], entry_zone_low) - 0.35 * atr_v
    take_profit = max(ind["recent_high_20"], ind["bb_upper"], ind["breakout_trigger"])
    if take_profit <= entry_zone_high:
        take_profit = entry_zone_high + 2.2 * atr_v

    return finalize_plan(
        symbol, interval, latest_price, ind, env_score,
        "trend_pullback", "趋势回踩做多",
        watch_zone_low, watch_zone_high,
        entry_zone_low, entry_zone_high,
        stop_loss, take_profit,
        cfg
    )

def build_oversold_bounce_plan(symbol: str, interval: str, latest_price: float, ind: dict, env_score: int, cfg: dict):
    atr_v = max(ind["atr14"], latest_price * 0.002)
    base = min(ind["bb_lower"], ind["recent_low_10"])

    watch_zone_low = base + 0.20 * atr_v
    watch_zone_high = base + 0.85 * atr_v

    entry_zone_low = base - 0.12 * atr_v
    entry_zone_high = base + 0.18 * atr_v

    stop_loss = entry_zone_low - 0.45 * atr_v
    take_profit = max(ind["bb_middle"], ind["ema20"], ind["recent_high_10"])
    if take_profit <= entry_zone_high:
        take_profit = entry_zone_high + 2.0 * atr_v

    plan = finalize_plan(
        symbol, interval, latest_price, ind, env_score,
        "oversold_bounce", "超跌反弹做多",
        watch_zone_low, watch_zone_high,
        entry_zone_low, entry_zone_high,
        stop_loss, take_profit,
        cfg
    )

    if ind["rsi14"] > 48 and latest_price > ind["bb_middle"]:
        plan["score"] = max(0, plan["score"] - 12)
        plan = apply_plan_grade(plan, latest_price, cfg)
        if plan["level"] == "entry":
            plan["level"] = "warning"
            plan["status"] = "预警"
            plan["action"] = "虽然在区间附近，但并非典型超跌环境，先观察。"
            plan["model_win_rate"] = max(35, plan["model_win_rate"] - 5)

    return plan

def build_range_bottom_plan(symbol: str, interval: str, latest_price: float, ind: dict, env_score: int, cfg: dict):
    atr_v = max(ind["atr14"], latest_price * 0.002)
    box_low = ind["recent_low_20"]
    box_high = ind["recent_high_20"]
    box_width = box_high - box_low

    if box_width <= 0:
        box_width = atr_v * 3

    watch_zone_low = box_low + 0.14 * box_width
    watch_zone_high = box_low + 0.28 * box_width

    entry_zone_low = box_low + 0.02 * box_width
    entry_zone_high = box_low + 0.14 * box_width

    stop_loss = box_low - 0.10 * box_width - 0.25 * atr_v
    take_profit = box_low + 0.72 * box_width
    if take_profit <= entry_zone_high:
        take_profit = entry_zone_high + 1.8 * atr_v

    plan = finalize_plan(
        symbol, interval, latest_price, ind, env_score,
        "range_bottom", "箱体下沿做多",
        watch_zone_low, watch_zone_high,
        entry_zone_low, entry_zone_high,
        stop_loss, take_profit,
        cfg
    )

    width_pct = box_width / max(latest_price, 1e-9) * 100.0
    if width_pct < 1.2 or width_pct > 15:
        plan["score"] = max(0, plan["score"] - 15)
        plan = apply_plan_grade(plan, latest_price, cfg)
        plan["model_win_rate"] = max(35, plan["model_win_rate"] - 6)

    return plan

# ======================================
# 单币评估
# ======================================
def evaluate_symbol(symbol: str, interval: str, latest_price: float, ind: dict, env_ind_4h: Optional[dict], cfg: dict):
    if interval == "1h" and env_ind_4h is not None:
        env_score = evaluate_environment(env_ind_4h)
    else:
        env_score = evaluate_environment(ind)

    plans = [
        build_trend_pullback_plan(symbol, interval, latest_price, ind, env_score, cfg),
        build_oversold_bounce_plan(symbol, interval, latest_price, ind, env_score, cfg),
        build_range_bottom_plan(symbol, interval, latest_price, ind, env_score, cfg),
    ]

    if latest_price < ind["recent_low_10"] and ind["macd_hist"] < ind["macd_hist_prev"] and ind["rsi14"] < 35:
        for p in plans:
            p["score"] = max(0, p["score"] - 10)
            p = apply_plan_grade(p, latest_price, cfg)
            if p["level"] == "entry":
                p["level"] = "warning"
                p["status"] = "预警"
                p["action"] = "虽然价格接近做多区，但跌势仍偏急，先等进一步止跌确认。"
                p["model_win_rate"] = max(35, p["model_win_rate"] - 5)

    best = sorted(
        plans,
        key=lambda x: (
            level_priority(x["level"]),
            x["score"],
            x["rr"],
            -PRIORITY[x["symbol"]],
        ),
        reverse=True
    )[0]

    return best

# ======================================
# 消息模板
# ======================================
def build_no_long_message(best: dict, dt_cn: datetime) -> str:
    symbol = short_symbol(best["symbol"])
    return (
        f"⛔【{symbol} {best['interval']}暂不做多】\n"
        f"时间（中国）：{fmt_time_full(dt_cn)}\n"
        f"当前价格：{fmt_price(best['symbol'], best['latest_price'])}\n"
        f"做多思路：{best['scenario_name']}\n"
        f"关注区间：{fmt_price(best['symbol'], best['watch_zone_low'])} - {fmt_price(best['symbol'], best['watch_zone_high'])}\n"
        f"做多区间：{fmt_price(best['symbol'], best['entry_zone_low'])} - {fmt_price(best['symbol'], best['entry_zone_high'])}\n"
        f"预计止损：{fmt_price(best['symbol'], best['stop_loss'])}\n"
        f"预计止盈：{fmt_price(best['symbol'], best['take_profit'])}\n"
        f"盈亏比：{best['rr']:.2f}\n"
        f"模型胜率：{best['model_win_rate']}%\n"
        f"评分：{best['score']}\n"
        f"结论：{best['action']}"
    )

def build_watch_message(best: dict, dt_cn: datetime) -> str:
    symbol = short_symbol(best["symbol"])
    return (
        f"⏸️【{symbol} {best['interval']}观望】\n"
        f"时间（中国）：{fmt_time_full(dt_cn)}\n"
        f"当前价格：{fmt_price(best['symbol'], best['latest_price'])}\n"
        f"做多思路：{best['scenario_name']}\n"
        f"预警区间：{fmt_price(best['symbol'], best['watch_zone_low'])} - {fmt_price(best['symbol'], best['watch_zone_high'])}\n"
        f"做多区间：{fmt_price(best['symbol'], best['entry_zone_low'])} - {fmt_price(best['symbol'], best['entry_zone_high'])}\n"
        f"预计止损：{fmt_price(best['symbol'], best['stop_loss'])}\n"
        f"预计止盈：{fmt_price(best['symbol'], best['take_profit'])}\n"
        f"盈亏比：{best['rr']:.2f}\n"
        f"模型胜率：{best['model_win_rate']}%\n"
        f"评分：{best['score']}\n"
        f"结论：{best['action']}"
    )

def build_warning_message(best: dict, dt_cn: datetime) -> str:
    symbol = short_symbol(best["symbol"])
    notes = "、".join(best["confirm_notes"][:3]) if best["confirm_notes"] else "确认还不够"
    overall_true_win_rate = get_overall_true_win_rate_text()

    return (
        f"⚠️【{symbol} {best['interval']}预警】\n"
        f"时间（中国）：{fmt_time_full(dt_cn)}\n"
        f"当前价格：{fmt_price(best['symbol'], best['latest_price'])}\n"
        f"做多思路：{best['scenario_name']}\n"
        f"预警区间：{fmt_price(best['symbol'], best['watch_zone_low'])} - {fmt_price(best['symbol'], best['watch_zone_high'])}\n"
        f"做多区间：{fmt_price(best['symbol'], best['entry_zone_low'])} - {fmt_price(best['symbol'], best['entry_zone_high'])}\n"
        f"预计止损：{fmt_price(best['symbol'], best['stop_loss'])}\n"
        f"预计止盈：{fmt_price(best['symbol'], best['take_profit'])}\n"
        f"预计盈亏比：{best['rr']:.2f}\n"
        f"模型胜率：{best['model_win_rate']}%\n"
        f"总体真实胜率：{overall_true_win_rate}\n"
        f"评分：{best['score']}\n"
        f"确认情况：{notes}\n"
        f"结论：{best['action']}"
    )

def build_entry_message(best: dict, dt_cn: datetime) -> str:
    symbol = short_symbol(best["symbol"])
    entry_price = best["latest_price"]
    position = suggest_position(best["score"], best["interval"])
    notes = "、".join(best["confirm_notes"][:4]) if best["confirm_notes"] else "已满足基础确认"
    overall_true_win_rate = get_overall_true_win_rate_text()

    return (
        f"🚨【{symbol} {best['interval']}开多信号】\n"
        f"时间（中国）：{fmt_time_full(dt_cn)}\n"
        f"当前价格：{fmt_price(best['symbol'], best['latest_price'])}\n"
        f"做多思路：{best['scenario_name']}\n"
        f"做多区间：{fmt_price(best['symbol'], best['entry_zone_low'])} - {fmt_price(best['symbol'], best['entry_zone_high'])}\n"
        f"建议开单价：{fmt_price(best['symbol'], entry_price)}\n"
        f"止损：{fmt_price(best['symbol'], best['stop_loss'])}\n"
        f"止盈：{fmt_price(best['symbol'], best['take_profit'])}\n"
        f"盈亏比：{best['rr']:.2f}\n"
        f"模型胜率：{best['model_win_rate']}%\n"
        f"总体真实胜率：{overall_true_win_rate}\n"
        f"评分：{best['score']}\n"
        f"仓位建议：{position}\n"
        f"确认情况：{notes}\n"
        f"结论：{best['action']}"
    )

# ======================================
# 表格行
# ======================================
def result_to_live_row(result: dict) -> dict:
    return {
        "symbol": short_symbol(result["symbol"]),
        "interval": result["interval"],
        "status": result["status"],
        "scenario_name": result["scenario_name"],
        "latest_price": fmt_price(result["symbol"], result["latest_price"]),
        "watch_zone": f"{fmt_price(result['symbol'], result['watch_zone_low'])} - {fmt_price(result['symbol'], result['watch_zone_high'])}",
        "entry_zone": f"{fmt_price(result['symbol'], result['entry_zone_low'])} - {fmt_price(result['symbol'], result['entry_zone_high'])}",
        "stop_loss": fmt_price(result["symbol"], result["stop_loss"]),
        "take_profit": fmt_price(result["symbol"], result["take_profit"]),
        "rr": f"{result['rr']:.2f}",
        "model_win_rate": f"{result['model_win_rate']}%",
        "overall_true_win_rate": get_overall_true_win_rate_text(),
        "score": str(result["score"]),
    }

# ======================================
# 单轮分析
# ======================================
def run_one_live_cycle():
    dt_cn = now_cn()

    latest_prices = {}
    price_errors = []

    for symbol in SYMBOLS:
        try:
            latest_prices[symbol] = fetch_latest_price(symbol)
        except Exception as e:
            price_errors.append(f"{symbol} 最新价获取失败：{str(e)}")

    indicator_cache = {}
    indicator_errors = []

    for symbol in SYMBOLS:
        if symbol not in latest_prices:
            continue

        for interval in INTERVALS:
            try:
                indicator_cache[(symbol, interval)] = calc_indicators_live(symbol, interval)
            except Exception as e:
                indicator_errors.append(f"{symbol} {interval} 指标失败：{str(e)}")

    all_results = []
    messages = []

    for interval in INTERVALS:
        interval_results = []
        interval_errors = []

        for symbol in SYMBOLS:
            if symbol not in latest_prices:
                interval_errors.append(f"{symbol} 缺少最新价，跳过 {interval}")
                continue

            ind = indicator_cache.get((symbol, interval))
            if ind is None:
                interval_errors.append(f"{symbol} 缺少 {interval} 指标")
                continue

            env_4h = indicator_cache.get((symbol, "4h"))

            try:
                result = evaluate_symbol(
                    symbol,
                    interval,
                    latest_prices[symbol],
                    ind,
                    env_4h,
                    STRATEGY_CONFIG
                )
                interval_results.append(result)
                all_results.append(result)
            except Exception as e:
                interval_errors.append(f"{symbol} {interval} 评估失败：{str(e)}")

        if not interval_results:
            text = (
                f"⚠️【{interval}分析失败】\n"
                f"时间（中国）：{fmt_time_full(dt_cn)}\n"
                f"本周期没有可用结果。"
            )
            if price_errors:
                text += "\n价格错误：\n" + "\n".join(price_errors[:8])
            if indicator_errors or interval_errors:
                text += "\n分析错误：\n" + "\n".join((indicator_errors + interval_errors)[:8])

            messages.append({
                "level": "warning",
                "interval": interval,
                "silent": False,
                "text": text,
            })
            continue

        best = sorted(
            interval_results,
            key=lambda x: (
                level_priority(x["level"]),
                x["score"],
                x["rr"],
                -PRIORITY[x["symbol"]],
            ),
            reverse=True
        )[0]

        if best["level"] == "entry":
            text = build_entry_message(best, dt_cn)
            silent = False   # 只有可开单才响
        elif best["level"] == "warning":
            text = build_warning_message(best, dt_cn)
            silent = True    # 预警改为静默
        elif best["level"] == "watch":
            text = build_watch_message(best, dt_cn)
            silent = True
        else:
            text = build_no_long_message(best, dt_cn)
            silent = True

        extra_errors = []
        if price_errors:
            extra_errors.extend(price_errors[:3])
        if interval_errors:
            extra_errors.extend(interval_errors[:3])

        if extra_errors:
            text += "\n\n附加异常：\n" + "\n".join(extra_errors)

        messages.append({
            "level": best["level"],
            "interval": interval,
            "silent": silent,
            "text": text,
        })

    rows = [result_to_live_row(x) for x in sorted(
        all_results,
        key=lambda r: (
            interval_rank(r["interval"]),
            -level_priority(r["level"]),
            -r["score"],
            PRIORITY[r["symbol"]],
        )
    )]

    print_live_desktop_report(rows, dt_cn)
    return messages

# ======================================
# 发送执行器
# ======================================
def send_item_with_strategy(item: dict):
    global LAST_PINNED_MESSAGE_ID

    if not should_send_message(item):
        print("-" * 100)
        print(f"跳过重复消息：{item['level']} {item['interval']}")
        return

    level = item["level"]
    text = item["text"]
    silent = item["silent"]

    first_message_id = None
    repeat_times = max(1, ENTRY_BURST_COUNT) if level == "entry" else 1
    send_success = False

    for i in range(repeat_times):
        try:
            resp = send_telegram_message(text=text, silent=silent)
            send_success = True

            if resp and isinstance(resp, dict):
                result = resp.get("result", {})
                if i == 0:
                    first_message_id = result.get("message_id")

            print("=" * 100)
            print(text)

        except Exception as e:
            print(f"发送 Telegram 消息失败：{str(e)}")

        if i < repeat_times - 1:
            time.sleep(ENTRY_BURST_GAP_SECONDS)

    if send_success:
        remember_sent_message(item)

    if level == "entry" and PIN_ENTRY_MESSAGE and first_message_id:
        try:
            if UNPIN_PREVIOUS_ENTRY_BEFORE_PIN and LAST_PINNED_MESSAGE_ID:
                if LAST_PINNED_MESSAGE_ID != first_message_id:
                    unpin_telegram_message(LAST_PINNED_MESSAGE_ID)

            pin_telegram_message(first_message_id, silent=PIN_SILENT)
            LAST_PINNED_MESSAGE_ID = first_message_id

        except Exception as e:
            print(f"置顶失败：{str(e)}")

# ======================================
# 主程序：执行一轮就退出
# ======================================
def main():
    messages = run_one_live_cycle()
    for item in messages:
        send_item_with_strategy(item)

if __name__ == "__main__":
    main()
