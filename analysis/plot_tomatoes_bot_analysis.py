from __future__ import annotations

import csv
import importlib.util
import sys
import types
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(r"C:\Users\lucas\OneDrive\Desktop\Coding\Prosperity")
TRADER_PATH = ROOT / "tomatoes_kelp_style_safe.py"


@dataclass
class Listing:
    symbol: str
    product: str
    denomination: str


@dataclass
class Order:
    symbol: str
    price: int
    quantity: int


@dataclass
class OrderDepth:
    buy_orders: Dict[int, int] = field(default_factory=dict)
    sell_orders: Dict[int, int] = field(default_factory=dict)


@dataclass
class Trade:
    symbol: str
    price: int
    quantity: int
    buyer: str = ""
    seller: str = ""
    timestamp: int = 0


@dataclass
class Observation:
    plainValueObservations: Dict = field(default_factory=dict)
    conversionObservations: Dict = field(default_factory=dict)


@dataclass
class TradingState:
    traderData: str
    timestamp: int
    listings: Dict[str, Listing]
    order_depths: Dict[str, OrderDepth]
    own_trades: Dict[str, List[Trade]]
    market_trades: Dict[str, List[Trade]]
    position: Dict[str, int]
    observations: Observation


def install_datamodel_stub() -> None:
    module = types.ModuleType("datamodel")
    module.Listing = Listing
    module.Order = Order
    module.OrderDepth = OrderDepth
    module.Trade = Trade
    module.Observation = Observation
    module.TradingState = TradingState
    sys.modules["datamodel"] = module


def load_trader():
    install_datamodel_stub()
    spec = importlib.util.spec_from_file_location("user_trader", TRADER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.Trader()


def parse_int(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    return int(float(value))


def load_prices(prices_path: Path) -> Dict[int, OrderDepth]:
    snapshots: Dict[int, OrderDepth] = {}
    with prices_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            if row["product"] != "TOMATOES":
                continue
            timestamp = int(row["timestamp"])
            buy_orders: Dict[int, int] = {}
            sell_orders: Dict[int, int] = {}
            for level in (1, 2, 3):
                bid_price = parse_int(row.get(f"bid_price_{level}", ""))
                bid_volume = parse_int(row.get(f"bid_volume_{level}", ""))
                ask_price = parse_int(row.get(f"ask_price_{level}", ""))
                ask_volume = parse_int(row.get(f"ask_volume_{level}", ""))
                if bid_price is not None and bid_volume is not None:
                    buy_orders[bid_price] = bid_volume
                if ask_price is not None and ask_volume is not None:
                    sell_orders[ask_price] = -ask_volume
            snapshots[timestamp] = OrderDepth(buy_orders=buy_orders, sell_orders=sell_orders)
    return snapshots


def best_bid_ask(depth: OrderDepth):
    best_bid = max(depth.buy_orders) if depth.buy_orders else None
    best_ask = min(depth.sell_orders) if depth.sell_orders else None
    return best_bid, best_ask


def clone_depth(depth: OrderDepth) -> OrderDepth:
    return OrderDepth(dict(depth.buy_orders), dict(depth.sell_orders))


def load_market_trades(trades_path: Path) -> Dict[int, Dict[str, List[Trade]]]:
    snapshots: Dict[int, Dict[str, List[Trade]]] = {}
    with trades_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            timestamp = int(row["timestamp"])
            symbol = row["symbol"]
            trade = Trade(
                symbol=symbol,
                price=int(float(row["price"])),
                quantity=int(float(row["quantity"])),
                buyer=row.get("buyer", "") or "",
                seller=row.get("seller", "") or "",
                timestamp=timestamp,
            )
            snapshots.setdefault(timestamp, {}).setdefault(symbol, []).append(trade)
    return snapshots


def match_order_depth(order: Order, depth: OrderDepth):
    fills = []
    remaining = abs(order.quantity)
    if order.quantity > 0:
        for ask_price in sorted(depth.sell_orders):
            available = -depth.sell_orders[ask_price]
            if ask_price > order.price or remaining <= 0 or available <= 0:
                continue
            fill_qty = min(remaining, available)
            fills.append((ask_price, fill_qty))
            depth.sell_orders[ask_price] += fill_qty
            remaining -= fill_qty
    else:
        for bid_price in sorted(depth.buy_orders, reverse=True):
            available = depth.buy_orders[bid_price]
            if bid_price < order.price or remaining <= 0 or available <= 0:
                continue
            fill_qty = min(remaining, available)
            fills.append((bid_price, -fill_qty))
            depth.buy_orders[bid_price] -= fill_qty
            remaining -= fill_qty
    return fills


def match_market_trades(order: Order, trades: List[Trade], best_bid: int, best_ask: int, fill_ratio: float):
    fills = []
    remaining = abs(order.quantity)
    if remaining == 0 or fill_ratio <= 0:
        return fills
    if order.quantity > 0 and order.price <= best_bid:
        return fills
    if order.quantity < 0 and order.price >= best_ask:
        return fills
    for trade in trades:
        if remaining <= 0:
            break
        visible_qty = max(0, int(trade.quantity * fill_ratio))
        if visible_qty <= 0:
            continue
        if order.quantity > 0 and trade.price <= order.price:
            fill_qty = min(remaining, visible_qty)
            fills.append((order.price, fill_qty))
            trade.quantity -= fill_qty
            remaining -= fill_qty
        elif order.quantity < 0 and trade.price >= order.price:
            fill_qty = min(remaining, visible_qty)
            fills.append((order.price, -fill_qty))
            trade.quantity -= fill_qty
            remaining -= fill_qty
    return fills


def moving_average(values: List[float], window: int) -> float:
    sample = values[-window:] if len(values) >= window else values
    return sum(sample) / len(sample)


def returns(values: List[float]) -> List[float]:
    return [values[i] - values[i - 1] for i in range(1, len(values))]


def mean_abs(values: List[float], window: int) -> float:
    if not values:
        return 0.0
    sample = values[-window:] if len(values) >= window else values
    return sum(abs(v) for v in sample) / len(sample)


def flip_ratio(values: List[float], window: int) -> float:
    if len(values) < 2:
        return 0.0
    sample = values[-window:] if len(values) >= window else values
    flips = 0
    pairs = 0
    for prev, cur in zip(sample, sample[1:]):
        if prev == 0 or cur == 0:
            continue
        pairs += 1
        if prev * cur < 0:
            flips += 1
    return flips / pairs if pairs else 0.0


def detect_regime(hist: List[float]) -> str:
    FAST_WINDOW = 12
    SLOW_WINDOW = 48
    VOL_WINDOW = 20
    LONG_VOL_WINDOW = 80
    FLIP_WINDOW = 24
    TREND_THRESHOLD = 3.2
    RESIDUAL_THRESHOLD = 2.5
    HIGH_VOL_RATIO = 2.2
    HIGH_FLIP_RATIO = 0.42

    if len(hist) < 5:
        return "warmup"
    fast_ma = moving_average(hist, FAST_WINDOW)
    slow_ma = moving_average(hist, SLOW_WINDOW)
    rets = returns(hist)
    short_vol = mean_abs(rets, VOL_WINDOW)
    long_vol = max(0.1, mean_abs(rets, LONG_VOL_WINDOW))
    flip = flip_ratio(rets, FLIP_WINDOW)
    trend = fast_ma - slow_ma
    residual = hist[-1] - slow_ma
    vol_ratio = short_vol / long_vol if long_vol > 0 else 1.0

    if vol_ratio >= HIGH_VOL_RATIO:
        return "high_vol"
    if trend >= TREND_THRESHOLD and flip < HIGH_FLIP_RATIO:
        return "trend_up"
    if trend <= -TREND_THRESHOLD and flip < HIGH_FLIP_RATIO:
        return "trend_down"
    if abs(residual) >= RESIDUAL_THRESHOLD and flip >= HIGH_FLIP_RATIO:
        return "mean_revert"
    return "neutral"


def contiguous_regions(timestamps: List[int], regimes: List[str]):
    if not timestamps:
        return []
    blocks = []
    start = timestamps[0]
    current = regimes[0]
    prev = timestamps[0]
    for ts, regime in zip(timestamps[1:], regimes[1:]):
        if regime != current:
            blocks.append((start, prev, current))
            start = ts
            current = regime
        prev = ts
    blocks.append((start, prev, current))
    return blocks


def analyze_day(day: int) -> Path:
    trader = load_trader()
    prices_path = ROOT / f"prices_round_0_day_{day}.csv"
    trades_path = ROOT / f"trades_round_0_day_{day}.csv"
    price_snapshots = load_prices(prices_path)
    trade_snapshots = load_market_trades(trades_path) if trades_path.exists() else {}

    timestamps = sorted(price_snapshots)
    listings = {"TOMATOES": Listing("TOMATOES", "TOMATOES", "SEASHELLS")}
    positions: Dict[str, int] = defaultdict(int)
    cash: Dict[str, float] = defaultdict(float)
    own_trades_prev: Dict[str, List[Trade]] = {}

    mids: List[float] = []
    best_bids: List[int] = []
    best_asks: List[int] = []
    regimes: List[str] = []
    buy_fill_ts: List[int] = []
    buy_fill_px: List[int] = []
    sell_fill_ts: List[int] = []
    sell_fill_px: List[int] = []
    position_hist: List[int] = []
    pnl_hist: List[float] = []
    regime_hist_source: List[float] = []

    for timestamp in timestamps:
        depth = price_snapshots[timestamp]
        best_bid, best_ask = best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            continue
        mid = 0.5 * (best_bid + best_ask)
        regime_hist_source.append(mid)
        regimes.append(detect_regime(regime_hist_source))
        mids.append(mid)
        best_bids.append(best_bid)
        best_asks.append(best_ask)

        market_trades = trade_snapshots.get(timestamp, {}).get("TOMATOES", [])
        working_depth = clone_depth(depth)
        working_trades = [
            Trade(
                symbol=trade.symbol,
                price=trade.price,
                quantity=trade.quantity,
                buyer=trade.buyer,
                seller=trade.seller,
                timestamp=trade.timestamp,
            )
            for trade in market_trades
        ]
        state = TradingState(
            traderData="",
            timestamp=timestamp,
            listings=listings,
            order_depths={"TOMATOES": depth},
            own_trades=own_trades_prev,
            market_trades={"TOMATOES": market_trades},
            position=dict(positions),
            observations=Observation(),
        )
        output = trader.run(state)
        orders_by_symbol = output[0] if isinstance(output, tuple) else output
        orders = orders_by_symbol.get("TOMATOES", [])
        own_trades_current: Dict[str, List[Trade]] = {}
        for order in orders:
            depth_fills = match_order_depth(order, working_depth)
            filled_qty = sum(abs(fill_qty) for _, fill_qty in depth_fills)
            remaining = abs(order.quantity) - filled_qty
            trade_fills = []
            if remaining > 0:
                residual = Order(order.symbol, order.price, remaining if order.quantity > 0 else -remaining)
                trade_fills = match_market_trades(
                    residual,
                    working_trades,
                    best_bid,
                    best_ask,
                    0.35,
                )
            fills = depth_fills + trade_fills
            for fill_price, fill_qty in fills:
                positions["TOMATOES"] += fill_qty
                cash["TOMATOES"] -= fill_qty * fill_price
                own_trades_current.setdefault("TOMATOES", []).append(
                    Trade(
                        symbol="TOMATOES",
                        price=fill_price,
                        quantity=abs(fill_qty),
                        buyer="SUBMISSION" if fill_qty > 0 else "",
                        seller="SUBMISSION" if fill_qty < 0 else "",
                        timestamp=timestamp,
                    )
                )
                if fill_qty > 0:
                    buy_fill_ts.append(timestamp)
                    buy_fill_px.append(fill_price)
                else:
                    sell_fill_ts.append(timestamp)
                    sell_fill_px.append(fill_price)

        own_trades_prev = own_trades_current
        position_hist.append(positions["TOMATOES"])
        pnl_hist.append(cash["TOMATOES"] + positions["TOMATOES"] * mid)

    colors = {
        "warmup": "#f0f0f0",
        "neutral": "#e8f1fb",
        "mean_revert": "#e6f6e6",
        "trend_up": "#fff2cc",
        "trend_down": "#f4cccc",
        "high_vol": "#eadcf8",
    }

    fig, (ax_price, ax_pos) = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    for start, end, regime in contiguous_regions(timestamps, regimes):
        ax_price.axvspan(start, end, color=colors.get(regime, "#ffffff"), alpha=0.35)

    ax_price.plot(timestamps, mids, color="black", linewidth=1.2, label="Mid")
    ax_price.plot(timestamps, best_bids, color="#2b6cff", linewidth=0.8, alpha=0.75, label="Best Bid")
    ax_price.plot(timestamps, best_asks, color="#d63b3b", linewidth=0.8, alpha=0.75, label="Best Ask")
    ax_price.scatter(buy_fill_ts, buy_fill_px, marker="^", color="#1a8f3c", s=36, label="Buy Fill", zorder=5)
    ax_price.scatter(sell_fill_ts, sell_fill_px, marker="v", color="#d97706", s=36, label="Sell Fill", zorder=5)
    ax_price.set_title(f"TOMATOES Day {day}: Price, Fills, and Regimes")
    ax_price.set_ylabel("Price")
    ax_price.grid(alpha=0.2)

    regime_handles = [Patch(facecolor=color, alpha=0.35, label=name.replace("_", " ").title()) for name, color in colors.items() if name != "warmup"]
    handles, labels = ax_price.get_legend_handles_labels()
    ax_price.legend(handles + regime_handles, labels + [h.get_label() for h in regime_handles], loc="upper left", ncol=3, fontsize=9)

    ax_pos.plot(timestamps, position_hist, color="#6a1b9a", linewidth=1.2, label="Position")
    ax_pos2 = ax_pos.twinx()
    ax_pos2.plot(timestamps, pnl_hist, color="#111111", linewidth=1.0, alpha=0.85, label="PnL")
    ax_pos.set_ylabel("Position")
    ax_pos2.set_ylabel("PnL")
    ax_pos.set_xlabel("Timestamp")
    ax_pos.grid(alpha=0.2)

    out_path = ROOT / f"tomatoes_day_{day}_analysis.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    for day in (-1, -2):
        path = analyze_day(day)
        print(path)


if __name__ == "__main__":
    main()
