from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List


class Trader:
    FAIR_VALUE = 10000
    POSITION_LIMIT = 60
    BASE_SIZE = 12
    FLATTEN_THRESHOLD = 30

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

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        depth = state.order_depths.get("ASH_COATED_OSMIUM")
        if depth is None:
            return result, 0, ""

        best_bid, bid_vol, best_ask, ask_vol = self.best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return result, 0, ""

        position = state.position.get("ASH_COATED_OSMIUM", 0)
        orders: List[Order] = []

        if best_ask < self.FAIR_VALUE:
            qty = self.clip_buy(min(ask_vol, self.BASE_SIZE), position)
            if qty > 0:
                orders.append(Order("ASH_COATED_OSMIUM", best_ask, qty))
                position += qty

        if best_bid > self.FAIR_VALUE:
            qty = self.clip_sell(min(bid_vol, self.BASE_SIZE), position)
            if qty > 0:
                orders.append(Order("ASH_COATED_OSMIUM", best_bid, -qty))
                position -= qty

        if position <= -self.FLATTEN_THRESHOLD:
            qty = self.clip_buy(min(-position, self.BASE_SIZE), position)
            if qty > 0:
                orders.append(Order("ASH_COATED_OSMIUM", self.FAIR_VALUE, qty))
                position += qty
        elif position >= self.FLATTEN_THRESHOLD:
            qty = self.clip_sell(min(position, self.BASE_SIZE), position)
            if qty > 0:
                orders.append(Order("ASH_COATED_OSMIUM", self.FAIR_VALUE, -qty))
                position -= qty

        bid_quote = min(best_bid + 1, self.FAIR_VALUE - 1)
        ask_quote = max(best_ask - 1, self.FAIR_VALUE + 1)

        buy_qty = self.clip_buy(self.BASE_SIZE, position)
        sell_qty = self.clip_sell(self.BASE_SIZE, position)

        if buy_qty > 0 and bid_quote < self.FAIR_VALUE:
            orders.append(Order("ASH_COATED_OSMIUM", bid_quote, buy_qty))
        if sell_qty > 0 and ask_quote > self.FAIR_VALUE:
            orders.append(Order("ASH_COATED_OSMIUM", ask_quote, -sell_qty))

        result["ASH_COATED_OSMIUM"] = orders
        return result, 0, ""
