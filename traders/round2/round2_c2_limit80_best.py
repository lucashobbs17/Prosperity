from datamodel import Order, OrderDepth, Trade, TradingState
from typing import Dict, List


class Trader:
    OSMIUM = "ASH_COATED_OSMIUM"
    PEPPER = "INTARIAN_PEPPER_ROOT"

    OSMIUM_FAIR_VALUE = 10000
    OSMIUM_POSITION_LIMIT = 80
    OSMIUM_BASE_SIZE = 12
    OSMIUM_FLATTEN_THRESHOLD = 36

    PEPPER_POSITION_LIMIT = 60
    PEPPER_BASE_SIZE = 10
    PEPPER_TARGET_LONG = 56
    PEPPER_FLATTEN_SHORT = -8
    PEPPER_FLATTEN_LONG = 64
    PEPPER_HISTORY_WINDOW = 24
    PEPPER_DRIFT_PER_TIMESTAMP = 0.001
    PEPPER_LARGE_TRADE_THRESHOLD = 7

    # Trained on round-1 days -2 and -1; still structurally consistent on round 2.
    PEPPER_ALPHA_INTERCEPT = 2.8
    PEPPER_ALPHA_FAIR_GAP = 1.45
    PEPPER_ALPHA_FAST_SLOW = -0.013530
    PEPPER_ALPHA_FAST_OLDER = -0.020296
    PEPPER_ALPHA_IMBALANCE = 0.48
    PEPPER_ALPHA_SPREAD = -0.002052
    PEPPER_ALPHA_TRADE_SIGNAL = 0.08

    def __init__(self):
        self.PEPPER_FAIR_BLEND = 0.75
        self.PEPPER_BUY_EDGE = 1.6
        self.PEPPER_SELL_EDGE = 4.6
        self.PEPPER_ALPHA_POS_TH1 = 0.5
        self.PEPPER_ALPHA_POS_TH2 = 1.2
        self.PEPPER_ALPHA_NEG_TH1 = -0.6
        self.PEPPER_ALPHA_STRONG_TH = 1.0
        self.PEPPER_ALPHA_VERY_NEG_TH = -0.8

        self.PEPPER_TARGET_MIN = 8
        self.PEPPER_TARGET_CAP = 65
        self.PEPPER_TARGET_BOOST_LOW = 4
        self.PEPPER_TARGET_BOOST_HIGH = 8
        self.PEPPER_TARGET_CUT = -4

        self.PEPPER_BUY_EXTRA = 12
        self.PEPPER_BUY_STRONG_EXTRA = 5
        self.PEPPER_SHORT_COVER_EXTRA = 10
        self.PEPPER_SHORT_COVER_SIZE = 8
        self.PEPPER_LONG_FLATTEN_SIZE = 6
        self.PEPPER_ASK_FLOOR_STRONG = 12
        self.PEPPER_ASK_FLOOR_WEAK = 8
        self.PEPPER_ASK_FLOOR_BASE = 4

    def best_bid_ask(self, depth: OrderDepth):
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        bid_vol = depth.buy_orders.get(best_bid, 0) if best_bid is not None else 0
        ask_vol = -depth.sell_orders.get(best_ask, 0) if best_ask is not None else 0
        return best_bid, bid_vol, best_ask, ask_vol

    def walls(self, depth: OrderDepth):
        bid_wall = min(depth.buy_orders) if depth.buy_orders else None
        ask_wall = max(depth.sell_orders) if depth.sell_orders else None
        wall_mid = None
        if bid_wall is not None and ask_wall is not None:
            wall_mid = 0.5 * (bid_wall + ask_wall)
        return bid_wall, wall_mid, ask_wall

    def improved_quotes(self, depth: OrderDepth, fair_bid: int, fair_ask: int):
        bid_wall, _, ask_wall = self.walls(depth)
        best_bid, _, best_ask, _ = self.best_bid_ask(depth)
        bid_quote = min(best_bid + 1, fair_bid - 1) if best_bid is not None else fair_bid - 1
        ask_quote = max(best_ask - 1, fair_ask + 1) if best_ask is not None else fair_ask + 1

        if bid_wall is not None:
            bid_quote = max(bid_quote, bid_wall + 1)
        if ask_wall is not None:
            ask_quote = min(ask_quote, ask_wall - 1)

        return bid_quote, ask_quote

    def clip_buy(self, qty: int, position: int, limit: int) -> int:
        return max(0, min(qty, limit - position))

    def clip_sell(self, qty: int, position: int, limit: int) -> int:
        return max(0, min(qty, limit + position))

    def load_shared_state(self, trader_data: str):
        if not trader_data:
            return {}
        parts = trader_data.split("|")
        state = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            state[key] = value
        return state

    def dump_shared_state(self, state: Dict[str, str]) -> str:
        return "|".join(f"{key}={value}" for key, value in state.items() if value)

    def infer_pepper_base(self, mid: int) -> int:
        return int(round(mid / 1000.0)) * 1000

    def load_pepper_history(self, raw: str, mid: float):
        if not raw:
            return self.infer_pepper_base(int(round(mid))), []
        parts = raw.split(";")
        try:
            base = int(parts[0]) if parts and parts[0] else self.infer_pepper_base(int(round(mid)))
        except ValueError:
            base = self.infer_pepper_base(int(round(mid)))
        try:
            history = [float(value) for value in parts[1].split(",") if value] if len(parts) > 1 else []
        except ValueError:
            history = []
        return base, history

    def dump_pepper_history(self, base: int, history: List[float]) -> str:
        payload = ",".join(f"{value:.1f}" for value in history[-self.PEPPER_HISTORY_WINDOW :])
        return f"{base};{payload}"

    def average(self, values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def pepper_fair_value(self, base: int, timestamp: int) -> float:
        return base + self.PEPPER_DRIFT_PER_TIMESTAMP * timestamp

    def pepper_trade_signal(self, trades: List[Trade], mid: float) -> float:
        total = 0.0
        for trade in trades:
            if trade.quantity < self.PEPPER_LARGE_TRADE_THRESHOLD:
                continue
            if trade.price > mid:
                total += trade.quantity
            elif trade.price < mid:
                total -= trade.quantity
        return total

    def pepper_linear_alpha(
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
            self.PEPPER_ALPHA_INTERCEPT
            + self.PEPPER_ALPHA_FAIR_GAP * (fair - mid)
            + self.PEPPER_ALPHA_FAST_SLOW * (fast - slow)
            + self.PEPPER_ALPHA_FAST_OLDER * (fast - older)
            + self.PEPPER_ALPHA_IMBALANCE * imbalance
            + self.PEPPER_ALPHA_SPREAD * spread
            + self.PEPPER_ALPHA_TRADE_SIGNAL * trade_signal
        )

    def osmium_orders(self, depth: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        best_bid, bid_vol, best_ask, ask_vol = self.best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        bid_wall, wall_mid, ask_wall = self.walls(depth)
        fair = self.OSMIUM_FAIR_VALUE if wall_mid is None else 0.82 * self.OSMIUM_FAIR_VALUE + 0.18 * wall_mid

        if best_ask <= fair - 1:
            qty = self.clip_buy(min(ask_vol, self.OSMIUM_BASE_SIZE), position, self.OSMIUM_POSITION_LIMIT)
            if qty > 0:
                orders.append(Order(self.OSMIUM, best_ask, qty))
                position += qty

        if best_bid >= fair + 1:
            qty = self.clip_sell(min(bid_vol, self.OSMIUM_BASE_SIZE), position, self.OSMIUM_POSITION_LIMIT)
            if qty > 0:
                orders.append(Order(self.OSMIUM, best_bid, -qty))
                position -= qty

        if position <= -self.OSMIUM_FLATTEN_THRESHOLD:
            qty = self.clip_buy(min(-position, self.OSMIUM_BASE_SIZE), position, self.OSMIUM_POSITION_LIMIT)
            if qty > 0:
                orders.append(Order(self.OSMIUM, self.OSMIUM_FAIR_VALUE, qty))
                position += qty
        elif position >= self.OSMIUM_FLATTEN_THRESHOLD:
            qty = self.clip_sell(min(position, self.OSMIUM_BASE_SIZE), position, self.OSMIUM_POSITION_LIMIT)
            if qty > 0:
                orders.append(Order(self.OSMIUM, self.OSMIUM_FAIR_VALUE, -qty))
                position -= qty

        bid_quote, ask_quote = self.improved_quotes(depth, int(fair), int(fair + 0.999999))

        if best_bid < fair and bid_wall is not None:
            for bp, bv in depth.buy_orders.items():
                if bp + 1 < fair and abs(bv) > 2:
                    bid_quote = max(bid_quote, bp + 1)
                    break
        if best_ask > fair and ask_wall is not None:
            for ap, av in depth.sell_orders.items():
                if ap - 1 > fair and abs(av) > 2:
                    ask_quote = min(ask_quote, ap - 1)
                    break

        buy_qty = self.clip_buy(self.OSMIUM_BASE_SIZE, position, self.OSMIUM_POSITION_LIMIT)
        sell_qty = self.clip_sell(self.OSMIUM_BASE_SIZE, position, self.OSMIUM_POSITION_LIMIT)

        if buy_qty > 0 and bid_quote < best_ask:
            orders.append(Order(self.OSMIUM, bid_quote, buy_qty))
        if sell_qty > 0 and ask_quote > best_bid:
            orders.append(Order(self.OSMIUM, ask_quote, -sell_qty))

        return orders

    def pepper_orders(self, state: TradingState, shared_state: Dict[str, str]) -> List[Order]:
        orders: List[Order] = []
        depth = state.order_depths.get(self.PEPPER)
        if depth is None:
            return orders

        best_bid, bid_vol, best_ask, ask_vol = self.best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        mid = 0.5 * (best_bid + best_ask)
        base, history = self.load_pepper_history(shared_state.get("pepper", ""), mid)
        history.append(mid)
        history = history[-self.PEPPER_HISTORY_WINDOW :]

        fair = self.pepper_fair_value(base, state.timestamp)
        fast = self.average(history[-8:])
        slow = self.average(history[-24:])
        older_slice = history[-24:-8] if len(history) >= 16 else history
        older = self.average(older_slice)
        spread = best_ask - best_bid
        _, wall_mid, _ = self.walls(depth)
        imbalance_denom = bid_vol + ask_vol
        imbalance = ((bid_vol - ask_vol) / imbalance_denom) if imbalance_denom else 0.0
        trade_signal = self.pepper_trade_signal(state.market_trades.get(self.PEPPER, []), mid)
        alpha = self.pepper_linear_alpha(fair, mid, fast, slow, older, imbalance, spread, trade_signal)

        fair_adjusted = fair + alpha
        if wall_mid is not None:
            fair_adjusted = self.PEPPER_FAIR_BLEND * fair_adjusted + (1 - self.PEPPER_FAIR_BLEND) * wall_mid
        fair_bid = int(fair_adjusted)
        fair_ask = int(fair_adjusted + 0.999999)

        target_long = self.PEPPER_TARGET_LONG + (
            self.PEPPER_TARGET_BOOST_HIGH if alpha > self.PEPPER_ALPHA_POS_TH2
            else self.PEPPER_TARGET_BOOST_LOW if alpha > self.PEPPER_ALPHA_POS_TH1
            else self.PEPPER_TARGET_CUT if alpha < self.PEPPER_ALPHA_NEG_TH1
            else 0
        )
        target_long = max(self.PEPPER_TARGET_MIN, min(target_long, self.PEPPER_TARGET_CAP))
        position = state.position.get(self.PEPPER, 0)

        if best_ask <= fair_adjusted + self.PEPPER_BUY_EDGE:
            buy_size = self.PEPPER_BASE_SIZE + (self.PEPPER_BUY_EXTRA if position < target_long else 0) + (self.PEPPER_BUY_STRONG_EXTRA if alpha > self.PEPPER_ALPHA_STRONG_TH else 0)
            qty = self.clip_buy(min(ask_vol, buy_size), position, self.PEPPER_POSITION_LIMIT)
            if qty > 0:
                orders.append(Order(self.PEPPER, best_ask, qty))
                position += qty

        if best_bid >= fair_adjusted + self.PEPPER_SELL_EDGE and position > target_long + 6:
            sell_size = max(4, self.PEPPER_BASE_SIZE - 2) + (self.PEPPER_LONG_FLATTEN_SIZE if alpha < self.PEPPER_ALPHA_VERY_NEG_TH else 0)
            qty = self.clip_sell(min(bid_vol, sell_size), position, self.PEPPER_POSITION_LIMIT)
            if qty > 0:
                orders.append(Order(self.PEPPER, best_bid, -qty))
                position -= qty

        if position <= self.PEPPER_FLATTEN_SHORT:
            qty = self.clip_buy(min(-position + self.PEPPER_SHORT_COVER_EXTRA, self.PEPPER_BASE_SIZE + self.PEPPER_SHORT_COVER_SIZE), position, self.PEPPER_POSITION_LIMIT)
            if qty > 0:
                orders.append(Order(self.PEPPER, fair_ask, qty))
                position += qty
        elif position >= self.PEPPER_FLATTEN_LONG:
            qty = self.clip_sell(min(position - target_long, self.PEPPER_BASE_SIZE + self.PEPPER_LONG_FLATTEN_SIZE), position, self.PEPPER_POSITION_LIMIT)
            if qty > 0:
                orders.append(Order(self.PEPPER, fair_bid, -qty))
                position -= qty

        bid_quote, ask_quote = self.improved_quotes(depth, fair_bid, fair_ask)
        ask_floor = fair_ask + (
            self.PEPPER_ASK_FLOOR_STRONG if alpha > self.PEPPER_ALPHA_STRONG_TH and position < target_long
            else self.PEPPER_ASK_FLOOR_WEAK if position < target_long
            else self.PEPPER_ASK_FLOOR_BASE
        )
        ask_quote = max(ask_quote, ask_floor)

        buy_qty = self.clip_buy(self.PEPPER_BASE_SIZE + (self.PEPPER_BUY_EXTRA if position < target_long else 0), position, self.PEPPER_POSITION_LIMIT)
        sell_qty = self.clip_sell(max(4, self.PEPPER_BASE_SIZE - 3), position, self.PEPPER_POSITION_LIMIT)

        if buy_qty > 0 and bid_quote < fair_adjusted:
            orders.append(Order(self.PEPPER, bid_quote, buy_qty))
        if sell_qty > 0 and position >= target_long + 4 and ask_quote > fair_adjusted:
            orders.append(Order(self.PEPPER, ask_quote, -sell_qty))

        shared_state["pepper"] = self.dump_pepper_history(base, history)
        return orders

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        shared_state = self.load_shared_state(state.traderData)

        if self.OSMIUM in state.order_depths:
            result[self.OSMIUM] = self.osmium_orders(
                state.order_depths[self.OSMIUM],
                state.position.get(self.OSMIUM, 0),
            )

        if self.PEPPER in state.order_depths:
            result[self.PEPPER] = self.pepper_orders(state, shared_state)

        return result, 0, self.dump_shared_state(shared_state)
