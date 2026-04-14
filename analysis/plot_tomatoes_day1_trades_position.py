from __future__ import annotations

import importlib.util
import sys
import types
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\lucas\OneDrive\Desktop\Coding\Prosperity")
PRICES_PATH = ROOT / "prices_round_0_day_-1.csv"
TRADES_PATH = ROOT / "trades_round_0_day_-1.csv"
TRADER_PATH = ROOT / "tomatoes_kelp_style_safe.py"
OUT_PATH = ROOT / "tomatoes_day_-1_trades_position.png"


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


def parse_int(value: str):
    value = (value or "").strip()
    if not value:
        return None
    return int(float(value))


def load_prices() -> Dict[int, OrderDepth]:
    import csv

    snapshots = {}
    with PRICES_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            if row["product"] != "TOMATOES":
                continue
            buy_orders = {}
            sell_orders = {}
            for level in (1, 2, 3):
                bid_price = parse_int(row.get(f"bid_price_{level}", ""))
                bid_volume = parse_int(row.get(f"bid_volume_{level}", ""))
                ask_price = parse_int(row.get(f"ask_price_{level}", ""))
                ask_volume = parse_int(row.get(f"ask_volume_{level}", ""))
                if bid_price is not None and bid_volume is not None:
                    buy_orders[bid_price] = bid_volume
                if ask_price is not None and ask_volume is not None:
                    sell_orders[ask_price] = -ask_volume
            snapshots[int(row["timestamp"])] = OrderDepth(buy_orders=buy_orders, sell_orders=sell_orders)
    return snapshots


def load_market_trades() -> Dict[int, List[Trade]]:
    import csv

    snapshots: Dict[int, List[Trade]] = {}
    with TRADES_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            if row["symbol"] != "TOMATOES":
                continue
            trade = Trade(
                symbol="TOMATOES",
                price=int(float(row["price"])),
                quantity=int(float(row["quantity"])),
                buyer=row.get("buyer", "") or "",
                seller=row.get("seller", "") or "",
                timestamp=int(row["timestamp"]),
            )
            snapshots.setdefault(trade.timestamp, []).append(trade)
    return snapshots


def best_bid_ask(depth: OrderDepth):
    best_bid = max(depth.buy_orders) if depth.buy_orders else None
    best_ask = min(depth.sell_orders) if depth.sell_orders else None
    return best_bid, best_ask


def clone_depth(depth: OrderDepth) -> OrderDepth:
    return OrderDepth(dict(depth.buy_orders), dict(depth.sell_orders))


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


def main() -> None:
    trader = load_trader()
    price_snapshots = load_prices()
    trade_snapshots = load_market_trades()

    timestamps = sorted(price_snapshots)
    listings = {"TOMATOES": Listing("TOMATOES", "TOMATOES", "SEASHELLS")}
    positions: Dict[str, int] = defaultdict(int)
    cash: Dict[str, float] = defaultdict(float)
    own_trades_prev: Dict[str, List[Trade]] = {}

    mids = []
    buy_fill_ts = []
    buy_fill_px = []
    sell_fill_ts = []
    sell_fill_px = []
    position_hist = []
    pnl_hist = []

    for timestamp in timestamps:
        depth = price_snapshots[timestamp]
        best_bid, best_ask = best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            continue
        mid = 0.5 * (best_bid + best_ask)
        mids.append(mid)

        working_depth = clone_depth(depth)
        working_trades = [
            Trade(
                symbol=t.symbol,
                price=t.price,
                quantity=t.quantity,
                buyer=t.buyer,
                seller=t.seller,
                timestamp=t.timestamp,
            )
            for t in trade_snapshots.get(timestamp, [])
        ]
        state = TradingState(
            traderData="",
            timestamp=timestamp,
            listings=listings,
            order_depths={"TOMATOES": depth},
            own_trades=own_trades_prev,
            market_trades={"TOMATOES": trade_snapshots.get(timestamp, [])},
            position=dict(positions),
            observations=Observation(),
        )
        output = trader.run(state)
        orders = (output[0] if isinstance(output, tuple) else output).get("TOMATOES", [])
        own_trades_current: Dict[str, List[Trade]] = {}

        for order in orders:
            depth_fills = match_order_depth(order, working_depth)
            filled_qty = sum(abs(qty) for _, qty in depth_fills)
            remaining = abs(order.quantity) - filled_qty
            trade_fills = []
            if remaining > 0:
                residual = Order(order.symbol, order.price, remaining if order.quantity > 0 else -remaining)
                trade_fills = match_market_trades(residual, working_trades, best_bid, best_ask, 0.35)
            for fill_price, fill_qty in depth_fills + trade_fills:
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

    fig, (ax_price, ax_pos) = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_price.plot(timestamps, mids, color="black", linewidth=1.4, label="Mid Price")
    ax_price.scatter(buy_fill_ts, buy_fill_px, marker="^", color="#16a34a", s=42, label="Buy")
    ax_price.scatter(sell_fill_ts, sell_fill_px, marker="v", color="#ea580c", s=42, label="Sell")
    ax_price.set_title("TOMATOES Day -1: Price with Entries and Exits")
    ax_price.set_ylabel("Price")
    ax_price.grid(alpha=0.2)
    ax_price.legend(loc="upper left")

    ax_pos.plot(timestamps, position_hist, color="#6a1b9a", linewidth=1.2, label="Position")
    ax_pos2 = ax_pos.twinx()
    ax_pos2.plot(timestamps, pnl_hist, color="#111111", linewidth=1.0, alpha=0.8, label="PnL")
    ax_pos.set_ylabel("Position")
    ax_pos2.set_ylabel("PnL")
    ax_pos.set_xlabel("Timestamp")
    ax_pos.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=160)
    plt.close(fig)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
