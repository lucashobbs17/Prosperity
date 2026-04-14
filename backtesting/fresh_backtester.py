from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import types
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


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


@dataclass
class Fill:
    symbol: str
    price: int
    quantity: int
    source: str


@dataclass
class DayData:
    round_id: int
    day_id: int
    prices_path: Path
    trades_path: Path | None


class BacktestError(Exception):
    pass


def install_datamodel_stub() -> None:
    module = types.ModuleType("datamodel")
    module.Listing = Listing
    module.Order = Order
    module.OrderDepth = OrderDepth
    module.Trade = Trade
    module.Observation = Observation
    module.TradingState = TradingState
    sys.modules["datamodel"] = module


def load_trader(trader_path: Path):
    install_datamodel_stub()
    spec = importlib.util.spec_from_file_location("user_trader", trader_path)
    if spec is None or spec.loader is None:
        raise BacktestError(f"Could not import trader from {trader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Trader"):
        raise BacktestError(f"{trader_path} does not define Trader")
    return module.Trader()


def parse_int(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    return int(float(value))


def parse_day_spec(spec: str) -> Tuple[int, int | None]:
    if "-" not in spec:
        return int(spec), None
    if "--" in spec:
        round_part, day_part = spec.split("--", 1)
        return int(round_part), -int(day_part)
    round_part, day_part = spec.split("-", 1)
    return int(round_part), int(day_part)


def parse_symbol_float_map(raw: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise BacktestError(
                f"Invalid symbol mapping '{part}'. Use SYMBOL=value,SYMBOL=value"
            )
        symbol, value = part.split("=", 1)
        values[symbol.strip()] = float(value.strip())
    return values


def discover_day_data(data_dir: Path, specs: Iterable[str]) -> List[DayData]:
    requested: List[Tuple[int, int | None]] = [parse_day_spec(spec) for spec in specs]
    available_prices = list(data_dir.glob("prices_round_*_day_*.csv"))
    if not available_prices:
        raise BacktestError(f"No prices files found in {data_dir}")

    discovered: List[DayData] = []
    for round_id, day_id in requested:
        matching = []
        for path in available_prices:
            stem = path.stem
            parts = stem.split("_")
            file_round = int(parts[2])
            file_day = int(parts[-1])
            if file_round != round_id:
                continue
            if day_id is None or file_day == day_id:
                matching.append((file_day, path))
        if not matching:
            raise BacktestError(f"No data found for spec {round_id if day_id is None else f'{round_id}-{day_id}'} in {data_dir}")
        for file_day, prices_path in sorted(matching):
            trades_path = data_dir / f"trades_round_{round_id}_day_{file_day}.csv"
            discovered.append(
                DayData(
                    round_id=round_id,
                    day_id=file_day,
                    prices_path=prices_path,
                    trades_path=trades_path if trades_path.exists() else None,
                )
            )
    return discovered


def load_prices(prices_path: Path) -> Dict[int, Dict[str, OrderDepth]]:
    snapshots: Dict[int, Dict[str, OrderDepth]] = {}
    with prices_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            timestamp = int(row["timestamp"])
            product = row["product"]
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
            snapshots.setdefault(timestamp, {})[product] = OrderDepth(
                buy_orders=buy_orders,
                sell_orders=sell_orders,
            )
    return snapshots


def load_market_trades(trades_path: Path | None) -> Dict[int, Dict[str, List[Trade]]]:
    snapshots: Dict[int, Dict[str, List[Trade]]] = {}
    if trades_path is None:
        return snapshots
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


def get_position_limits(trader) -> Dict[str, int]:
    limits = getattr(trader, "POSITION_LIMITS", None)
    if isinstance(limits, dict):
        return {str(symbol): int(limit) for symbol, limit in limits.items()}
    return {}


def best_bid_ask(depth: OrderDepth) -> Tuple[int | None, int | None]:
    best_bid = max(depth.buy_orders) if depth.buy_orders else None
    best_ask = min(depth.sell_orders) if depth.sell_orders else None
    return best_bid, best_ask


def mark_to_mid(depth: OrderDepth | None) -> float:
    if depth is None:
        return 0.0
    bid, ask = best_bid_ask(depth)
    if bid is not None and ask is not None:
        return 0.5 * (bid + ask)
    if bid is not None:
        return float(bid)
    if ask is not None:
        return float(ask)
    return 0.0


def validate_orders_for_limits(
    orders_by_symbol: Dict[str, List[Order]],
    positions: Dict[str, int],
    limits: Dict[str, int],
) -> Dict[str, List[Order]]:
    valid_orders: Dict[str, List[Order]] = {}
    for symbol, orders in orders_by_symbol.items():
        limit = limits.get(symbol)
        if limit is None:
            valid_orders[symbol] = orders
            continue
        current = positions.get(symbol, 0)
        gross_buy = sum(max(0, order.quantity) for order in orders)
        gross_sell = sum(max(0, -order.quantity) for order in orders)
        if current + gross_buy > limit or current - gross_sell < -limit:
            valid_orders[symbol] = []
        else:
            valid_orders[symbol] = orders
    return valid_orders


def clone_depth(depth: OrderDepth) -> OrderDepth:
    return OrderDepth(buy_orders=dict(depth.buy_orders), sell_orders=dict(depth.sell_orders))


def clone_trades(trades_by_symbol: Dict[str, List[Trade]]) -> Dict[str, List[Trade]]:
    cloned: Dict[str, List[Trade]] = {}
    for symbol, trades in trades_by_symbol.items():
        cloned[symbol] = [
            Trade(
                symbol=trade.symbol,
                price=trade.price,
                quantity=trade.quantity,
                buyer=trade.buyer,
                seller=trade.seller,
                timestamp=trade.timestamp,
            )
            for trade in trades
        ]
    return cloned


def match_order_depth(order: Order, depth: OrderDepth) -> List[Fill]:
    fills: List[Fill] = []
    remaining = abs(order.quantity)
    if remaining == 0:
        return fills

    if order.quantity > 0:
        for ask_price in sorted(depth.sell_orders):
            available = -depth.sell_orders[ask_price]
            if ask_price > order.price or remaining <= 0 or available <= 0:
                continue
            fill_qty = min(remaining, available)
            fills.append(Fill(order.symbol, ask_price, fill_qty, "order_depth"))
            depth.sell_orders[ask_price] += fill_qty
            remaining -= fill_qty
    else:
        for bid_price in sorted(depth.buy_orders, reverse=True):
            available = depth.buy_orders[bid_price]
            if bid_price < order.price or remaining <= 0 or available <= 0:
                continue
            fill_qty = min(remaining, available)
            fills.append(Fill(order.symbol, bid_price, -fill_qty, "order_depth"))
            depth.buy_orders[bid_price] -= fill_qty
            remaining -= fill_qty
    return fills


def match_market_trades(
    order: Order,
    trades: List[Trade],
    mode: str,
    best_bid: int | None,
    best_ask: int | None,
    fill_ratio: float,
) -> List[Fill]:
    fills: List[Fill] = []
    if mode == "none" or fill_ratio <= 0:
        return fills

    remaining = abs(order.quantity)
    if remaining == 0:
        return fills

    if order.quantity > 0 and best_bid is not None and order.price <= best_bid:
        return fills
    if order.quantity < 0 and best_ask is not None and order.price >= best_ask:
        return fills

    for trade in trades:
        if remaining <= 0:
            break
        if order.quantity > 0:
            eligible = trade.price <= order.price if mode == "all" else trade.price < order.price
            if not eligible:
                continue
            visible_qty = max(0, int(trade.quantity * fill_ratio))
            if visible_qty <= 0:
                continue
            fill_qty = min(remaining, visible_qty)
            fills.append(Fill(order.symbol, order.price, fill_qty, "market_trade"))
            trade.quantity -= fill_qty
            remaining -= fill_qty
        else:
            eligible = trade.price >= order.price if mode == "all" else trade.price > order.price
            if not eligible:
                continue
            visible_qty = max(0, int(trade.quantity * fill_ratio))
            if visible_qty <= 0:
                continue
            fill_qty = min(remaining, visible_qty)
            fills.append(Fill(order.symbol, order.price, -fill_qty, "market_trade"))
            trade.quantity -= fill_qty
            remaining -= fill_qty
    return fills


def resolve_trade_fill_ratio(
    symbol: str,
    default_ratio: float,
    per_symbol_ratio: Dict[str, float],
) -> float:
    return per_symbol_ratio.get(symbol, default_ratio)


def apply_fills(
    fills: List[Fill],
    positions: Dict[str, int],
    cash: Dict[str, float],
    own_trades: Dict[str, List[Trade]],
) -> None:
    for fill in fills:
        positions[fill.symbol] += fill.quantity
        cash[fill.symbol] -= fill.quantity * fill.price
        own_trades.setdefault(fill.symbol, []).append(
            Trade(
                symbol=fill.symbol,
                price=fill.price,
                quantity=abs(fill.quantity),
                buyer="SUBMISSION" if fill.quantity > 0 else "",
                seller="SUBMISSION" if fill.quantity < 0 else "",
            )
        )


def run_day(
    trader_path: Path,
    day_data: DayData,
    match_trades_mode: str,
    trade_fill_ratio: float,
    trade_fill_ratio_by_symbol: Dict[str, float],
) -> dict:
    trader = load_trader(trader_path)
    limits = get_position_limits(trader)
    price_snapshots = load_prices(day_data.prices_path)
    trade_snapshots = load_market_trades(day_data.trades_path)
    timestamps = sorted(price_snapshots)

    positions: Dict[str, int] = defaultdict(int)
    cash: Dict[str, float] = defaultdict(float)
    trader_data = ""
    own_trades_prev: Dict[str, List[Trade]] = {}
    listings: Dict[str, Listing] = {}
    stats = defaultdict(lambda: defaultdict(int))
    last_depths: Dict[str, OrderDepth] = {}

    os.environ["PROSPERITY3BT_ROUND"] = str(day_data.round_id)
    os.environ["PROSPERITY3BT_DAY"] = str(day_data.day_id)

    for timestamp in timestamps:
        order_depths = price_snapshots[timestamp]
        market_trades = trade_snapshots.get(timestamp, {})
        working_depths = {symbol: clone_depth(depth) for symbol, depth in order_depths.items()}
        working_trades = clone_trades(market_trades)
        for symbol in order_depths:
            listings.setdefault(symbol, Listing(symbol, symbol, "SEASHELLS"))

        state = TradingState(
            traderData=trader_data,
            timestamp=timestamp,
            listings=listings,
            order_depths=order_depths,
            own_trades=own_trades_prev,
            market_trades=market_trades,
            position=dict(positions),
            observations=Observation(),
        )

        output = trader.run(state)
        if isinstance(output, tuple) and len(output) == 3:
            orders_by_symbol, _, trader_data = output
        else:
            orders_by_symbol = output
            trader_data = ""

        if not isinstance(orders_by_symbol, dict):
            raise BacktestError(f"Trader returned unsupported result at timestamp {timestamp}")

        filtered_orders = validate_orders_for_limits(orders_by_symbol, positions, limits)
        own_trades_current: Dict[str, List[Trade]] = {}

        for symbol, orders in filtered_orders.items():
            depth = working_depths.get(symbol)
            if depth is None:
                continue
            day_market_trades = working_trades.get(symbol, [])
            for order in orders:
                stats[symbol]["orders_sent"] += 1
                stats[symbol]["quoted_volume"] += abs(order.quantity)
                best_bid, best_ask = best_bid_ask(order_depths[symbol])
                depth_fills = match_order_depth(order, depth)
                filled_qty = sum(abs(fill.quantity) for fill in depth_fills)
                trade_fills: List[Fill] = []
                remaining = abs(order.quantity) - filled_qty
                if remaining > 0:
                    residual = Order(order.symbol, order.price, remaining if order.quantity > 0 else -remaining)
                    symbol_trade_fill_ratio = resolve_trade_fill_ratio(
                        symbol,
                        trade_fill_ratio,
                        trade_fill_ratio_by_symbol,
                    )
                    trade_fills = match_market_trades(
                        residual,
                        day_market_trades,
                        match_trades_mode,
                        best_bid,
                        best_ask,
                        symbol_trade_fill_ratio,
                    )
                fills = depth_fills + trade_fills
                apply_fills(fills, positions, cash, own_trades_current)
                for fill in fills:
                    if fill.source == "order_depth":
                        if fill.quantity > 0:
                            stats[symbol]["depth_buys"] += fill.quantity
                        else:
                            stats[symbol]["depth_sells"] += -fill.quantity
                    else:
                        if fill.quantity > 0:
                            stats[symbol]["trade_buys"] += fill.quantity
                        else:
                            stats[symbol]["trade_sells"] += -fill.quantity

        own_trades_prev = own_trades_current
        last_depths = order_depths

    products: Dict[str, dict] = {}
    total_pnl = 0.0
    for symbol, depth in last_depths.items():
        mid = mark_to_mid(depth)
        position = positions[symbol]
        pnl = cash[symbol] + position * mid
        total_pnl += pnl
        products[symbol] = {
            "pnl": round(pnl, 2),
            "cash": round(cash[symbol], 2),
            "position": position,
            "mid": mid,
            "orders_sent": stats[symbol]["orders_sent"],
            "quoted_volume": stats[symbol]["quoted_volume"],
            "depth_buys": stats[symbol]["depth_buys"],
            "depth_sells": stats[symbol]["depth_sells"],
            "trade_buys": stats[symbol]["trade_buys"],
            "trade_sells": stats[symbol]["trade_sells"],
        }

    return {
        "round": day_data.round_id,
        "day": day_data.day_id,
        "prices": str(day_data.prices_path),
        "trades": str(day_data.trades_path) if day_data.trades_path else None,
        "total_pnl": round(total_pnl, 2),
        "products": products,
    }


def format_text_report(reports: List[dict], merge_pnl: bool) -> str:
    lines: List[str] = []
    if merge_pnl:
        merged_total = sum(report["total_pnl"] for report in reports)
        merged_products = defaultdict(float)
        for report in reports:
            for symbol, payload in report["products"].items():
                merged_products[symbol] += payload["pnl"]
        lines.append(f"Merged total PnL: {merged_total:.2f}")
        for symbol in sorted(merged_products):
            lines.append(f"  {symbol}: {merged_products[symbol]:.2f}")
        return "\n".join(lines)

    for report in reports:
        lines.append(f"Round {report['round']} day {report['day']}: total PnL {report['total_pnl']:.2f}")
        for symbol, payload in sorted(report["products"].items()):
            lines.append(
                "  "
                f"{symbol}: pnl={payload['pnl']:.2f} pos={payload['position']} mid={payload['mid']:.1f} "
                f"orders={payload['orders_sent']} depth=({payload['depth_buys']}/{payload['depth_sells']}) "
                f"trades=({payload['trade_buys']}/{payload['trade_sells']})"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fresh Prosperity backtester inspired by prosperity3bt.")
    parser.add_argument("trader", type=Path, help="Path to the trader file")
    parser.add_argument("specs", nargs="+", help="Round/day specs such as 0, 0--1, 0--2")
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(), help="Directory containing prices/trades CSV files")
    parser.add_argument("--match-trades", choices=("all", "worse", "none"), default="all")
    parser.add_argument("--trade-fill-ratio", type=float, default=0.35, help="Fraction of historical trade volume assumed capturable by passive orders")
    parser.add_argument(
        "--trade-fill-ratio-by-symbol",
        type=str,
        default="",
        help="Optional per-symbol trade fill ratios, e.g. EMERALDS=0.12,TOMATOES=0.65",
    )
    parser.add_argument("--merge-pnl", action="store_true", help="Print merged PnL across all requested days")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of text summary")
    parser.add_argument("--out", type=Path, help="Optional path to save the report")
    args = parser.parse_args()
    trade_fill_ratio_by_symbol = parse_symbol_float_map(args.trade_fill_ratio_by_symbol)

    reports = [
        run_day(
            args.trader.resolve(),
            day_data,
            args.match_trades,
            args.trade_fill_ratio,
            trade_fill_ratio_by_symbol,
        )
        for day_data in discover_day_data(args.data_dir.resolve(), args.specs)
    ]

    output = json.dumps(reports, indent=2) if args.json else format_text_report(reports, args.merge_pnl)
    print(output)

    if args.out:
        args.out.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
