from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
import os


class Trader:
    OSMIUM_LIMIT = 60
    PEPPER_LIMIT = 60

    OSMIUM_FAIR = 10000
    OSMIUM_BASE_SIZE = 12
    OSMIUM_FLATTEN_THRESHOLD = 30

    PEPPER_BASE_SIZE = 10
    PEPPER_FLATTEN_THRESHOLD = 28

    PEPPER_FAIR_BY_DAY = {
        -2: 10500,
        -1: 11500,
        0: 12500,
    }

    def best_bid_ask(self, depth: OrderDepth):
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        bid_vol = depth.buy_orders.get(best_bid, 0) if best_bid is not None else 0
        ask_vol = -depth.sell_orders.get(best_ask, 0) if best_ask is not None else 0
        return best_bid, bid_vol, best_ask, ask_vol

    def clip_buy(self, qty: int, position: int, limit: int) -> int:
        return max(0, min(qty, limit - position))

    def clip_sell(self, qty: int, position: int, limit: int) -> int:
        return max(0, min(qty, limit + position))

    def get_round_day(self) -> int:
        raw = os.getenv("PROSPERITY3BT_DAY", "0")
        try:
            return int(raw)
        except Exception:
            return 0

    def anchored_orders(
        self,
        symbol: str,
        depth: OrderDepth,
        position: int,
        fair: int,
        limit: int,
        base_size: int,
        flatten_threshold: int,
    ) -> List[Order]:
        orders: List[Order] = []
        best_bid, bid_vol, best_ask, ask_vol = self.best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        if best_ask < fair:
            qty = self.clip_buy(min(ask_vol, base_size), position, limit)
            if qty > 0:
                orders.append(Order(symbol, best_ask, qty))
                position += qty

        if best_bid > fair:
            qty = self.clip_sell(min(bid_vol, base_size), position, limit)
            if qty > 0:
                orders.append(Order(symbol, best_bid, -qty))
                position -= qty

        if position <= -flatten_threshold:
            qty = self.clip_buy(min(-position, base_size), position, limit)
            if qty > 0:
                orders.append(Order(symbol, fair, qty))
                position += qty
        elif position >= flatten_threshold:
            qty = self.clip_sell(min(position, base_size), position, limit)
            if qty > 0:
                orders.append(Order(symbol, fair, -qty))
                position -= qty

        bid_quote = min(best_bid + 1, fair - 1)
        ask_quote = max(best_ask - 1, fair + 1)

        buy_qty = self.clip_buy(base_size, position, limit)
        sell_qty = self.clip_sell(base_size, position, limit)

        if buy_qty > 0 and bid_quote < fair:
            orders.append(Order(symbol, bid_quote, buy_qty))
        if sell_qty > 0 and ask_quote > fair:
            orders.append(Order(symbol, ask_quote, -sell_qty))

        return orders

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        day = self.get_round_day()
        pepper_fair = self.PEPPER_FAIR_BY_DAY.get(day, 12500)

        if "ASH_COATED_OSMIUM" in state.order_depths:
            result["ASH_COATED_OSMIUM"] = self.anchored_orders(
                "ASH_COATED_OSMIUM",
                state.order_depths["ASH_COATED_OSMIUM"],
                state.position.get("ASH_COATED_OSMIUM", 0),
                self.OSMIUM_FAIR,
                self.OSMIUM_LIMIT,
                self.OSMIUM_BASE_SIZE,
                self.OSMIUM_FLATTEN_THRESHOLD,
            )

        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            result["INTARIAN_PEPPER_ROOT"] = self.anchored_orders(
                "INTARIAN_PEPPER_ROOT",
                state.order_depths["INTARIAN_PEPPER_ROOT"],
                state.position.get("INTARIAN_PEPPER_ROOT", 0),
                pepper_fair,
                self.PEPPER_LIMIT,
                self.PEPPER_BASE_SIZE,
                self.PEPPER_FLATTEN_THRESHOLD,
            )

        return result, 0, ""
