from datamodel import Order, OrderDepth, Trade, TradingState
from typing import Dict, List


class Trader:
    SYMBOL = "INTARIAN_PEPPER_ROOT"
    POSITION_LIMIT = 60
    BASE_SIZE = 10
    TARGET_LONG = 16
    FLATTEN_SHORT = -10
    FLATTEN_LONG = 48
    HISTORY_WINDOW = 24
    DRIFT_PER_TIMESTAMP = 0.001
    LARGE_TRADE_THRESHOLD = 7

    # Trained on days -2 and -1 using fit_pepper_linear_alpha.py.
    ALPHA_INTERCEPT = 1.128179
    ALPHA_FAIR_GAP = 0.966260
    ALPHA_FAST_SLOW = -0.013530
    ALPHA_FAST_OLDER = -0.020296
    ALPHA_IMBALANCE = 0.114197
    ALPHA_SPREAD = -0.002052
    ALPHA_TRADE_SIGNAL = 0.011219

    def best_bid_ask(self, depth: OrderDepth):
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        bid_vol = depth.buy_orders.get(best_bid, 0) if best_bid is not None else 0
        ask_vol = -depth.sell_orders.get(best_ask, 0) if best_ask is not None else 0
        return best_bid, bid_vol, best_ask, ask_vol

    def clip_buy(self, qty: int, position: int) -> int:
        return max(0, min(qty, self.POSITION_LIMIT - position))

    def clip_sell(self, qty: int, position: int) -> int:
        return max(0, min(qty, self.POSITION_LIMIT + position))

    def infer_base(self, mid: int) -> int:
        return int(round(mid / 1000.0)) * 1000

    def load_history(self, trader_data: str, mid: float):
        if not trader_data:
            return self.infer_base(int(round(mid))), []
        parts = trader_data.split(";")
        try:
            base = int(parts[0]) if parts and parts[0] else self.infer_base(int(round(mid)))
        except ValueError:
            base = self.infer_base(int(round(mid)))
        try:
            history = [float(value) for value in parts[1].split(",") if value] if len(parts) > 1 else []
        except ValueError:
            history = []
        return base, history

    def dump_history(self, base: int, history: List[float]) -> str:
        return f"{base};" + ",".join(f"{value:.1f}" for value in history[-self.HISTORY_WINDOW :])

    def fair_value(self, base: int, timestamp: int) -> float:
        return base + self.DRIFT_PER_TIMESTAMP * timestamp

    def average(self, values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def infer_trade_signal(self, trades: List[Trade], mid: float) -> float:
        total = 0.0
        for trade in trades:
            if trade.quantity < self.LARGE_TRADE_THRESHOLD:
                continue
            if trade.price > mid:
                total += trade.quantity
            elif trade.price < mid:
                total -= trade.quantity
        return total

    def linear_alpha(
        self,
        fair: float,
        mid: float,
        fast: float,
        slow: float,
        older: float,
        imbalance: float,
        spread: float,
        trade_signal: float,
    ) -> float:
        return (
            self.ALPHA_INTERCEPT
            + self.ALPHA_FAIR_GAP * (fair - mid)
            + self.ALPHA_FAST_SLOW * (fast - slow)
            + self.ALPHA_FAST_OLDER * (fast - older)
            + self.ALPHA_IMBALANCE * imbalance
            + self.ALPHA_SPREAD * spread
            + self.ALPHA_TRADE_SIGNAL * trade_signal
        )

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        depth = state.order_depths.get(self.SYMBOL)
        if depth is None:
            return result, 0, ""

        best_bid, bid_vol, best_ask, ask_vol = self.best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return result, 0, ""

        mid = 0.5 * (best_bid + best_ask)
        base, history = self.load_history(state.traderData, mid)
        history.append(mid)
        history = history[-self.HISTORY_WINDOW :]

        fair = self.fair_value(base, state.timestamp)
        fast = self.average(history[-8:])
        slow = self.average(history[-24:])
        older_slice = history[-24:-8] if len(history) >= 16 else history
        older = self.average(older_slice)
        spread = best_ask - best_bid
        imbalance_denom = bid_vol + ask_vol
        imbalance = ((bid_vol - ask_vol) / imbalance_denom) if imbalance_denom else 0.0
        trade_signal = self.infer_trade_signal(state.market_trades.get(self.SYMBOL, []), mid)
        alpha = self.linear_alpha(fair, mid, fast, slow, older, imbalance, spread, trade_signal)

        fair_adjusted = fair + alpha
        fair_bid = int(fair_adjusted)
        fair_ask = int(fair_adjusted + 0.999999)

        target_long = self.TARGET_LONG + (8 if alpha > 1.2 else 4 if alpha > 0.5 else -6 if alpha < -0.6 else 0)
        target_long = max(0, min(target_long, 34))
        position = state.position.get(self.SYMBOL, 0)
        orders: List[Order] = []

        if best_ask <= fair_adjusted + 0.8:
            buy_size = self.BASE_SIZE + (4 if position < target_long else 0) + (2 if alpha > 1.0 else 0)
            qty = self.clip_buy(min(ask_vol, buy_size), position)
            if qty > 0:
                orders.append(Order(self.SYMBOL, best_ask, qty))
                position += qty

        if best_bid >= fair_adjusted + 1.7 and position > target_long:
            sell_size = self.BASE_SIZE + (2 if alpha < -0.8 else 0)
            qty = self.clip_sell(min(bid_vol, sell_size), position)
            if qty > 0:
                orders.append(Order(self.SYMBOL, best_bid, -qty))
                position -= qty

        if position <= self.FLATTEN_SHORT:
            qty = self.clip_buy(min(-position + 6, self.BASE_SIZE + 4), position)
            if qty > 0:
                orders.append(Order(self.SYMBOL, fair_ask, qty))
                position += qty
        elif position >= self.FLATTEN_LONG:
            qty = self.clip_sell(min(position - target_long, self.BASE_SIZE + 2), position)
            if qty > 0:
                orders.append(Order(self.SYMBOL, fair_bid, -qty))
                position -= qty

        bid_quote = min(best_bid + 1, fair_bid - 1)
        ask_floor = fair_ask + (4 if alpha > 1.0 and position < target_long else 2 if position < target_long else 0)
        ask_quote = max(best_ask - 1, ask_floor)

        buy_qty = self.clip_buy(self.BASE_SIZE + (4 if position < target_long else 0), position)
        sell_qty = self.clip_sell(self.BASE_SIZE, position)

        if buy_qty > 0 and bid_quote < fair_adjusted:
            orders.append(Order(self.SYMBOL, bid_quote, buy_qty))
        if sell_qty > 0 and position >= target_long - 2 and ask_quote > fair_adjusted:
            orders.append(Order(self.SYMBOL, ask_quote, -sell_qty))

        result[self.SYMBOL] = orders
        return result, 0, self.dump_history(base, history)
