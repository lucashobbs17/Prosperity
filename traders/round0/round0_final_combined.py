from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List


class Trader:
    EM_FAIR_VALUE = 10000
    EM_POSITION_LIMIT = 60
    EM_BASE_SIZE = 12
    EM_FLATTEN_THRESHOLD = 30

    T_POSITION_LIMIT = 60
    T_BASE_SIZE = 8
    T_TAKE_SIZE = 10
    T_FLATTEN_THRESHOLD = 30

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

    def emeralds_orders(self, depth: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        best_bid, bid_vol, best_ask, ask_vol = self.best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        if best_ask < self.EM_FAIR_VALUE:
            qty = self.clip_buy(min(ask_vol, self.EM_BASE_SIZE), position, self.EM_POSITION_LIMIT)
            if qty > 0:
                orders.append(Order("EMERALDS", best_ask, qty))
                position += qty

        if best_bid > self.EM_FAIR_VALUE:
            qty = self.clip_sell(min(bid_vol, self.EM_BASE_SIZE), position, self.EM_POSITION_LIMIT)
            if qty > 0:
                orders.append(Order("EMERALDS", best_bid, -qty))
                position -= qty

        if position <= -self.EM_FLATTEN_THRESHOLD:
            qty = self.clip_buy(min(-position, self.EM_BASE_SIZE), position, self.EM_POSITION_LIMIT)
            if qty > 0:
                orders.append(Order("EMERALDS", self.EM_FAIR_VALUE, qty))
                position += qty
        elif position >= self.EM_FLATTEN_THRESHOLD:
            qty = self.clip_sell(min(position, self.EM_BASE_SIZE), position, self.EM_POSITION_LIMIT)
            if qty > 0:
                orders.append(Order("EMERALDS", self.EM_FAIR_VALUE, -qty))
                position -= qty

        bid_quote = min(best_bid + 1, self.EM_FAIR_VALUE - 1)
        ask_quote = max(best_ask - 1, self.EM_FAIR_VALUE + 1)

        buy_qty = self.clip_buy(self.EM_BASE_SIZE, position, self.EM_POSITION_LIMIT)
        sell_qty = self.clip_sell(self.EM_BASE_SIZE, position, self.EM_POSITION_LIMIT)

        if buy_qty > 0 and bid_quote < self.EM_FAIR_VALUE:
            orders.append(Order("EMERALDS", bid_quote, buy_qty))
        if sell_qty > 0 and ask_quote > self.EM_FAIR_VALUE:
            orders.append(Order("EMERALDS", ask_quote, -sell_qty))

        return orders

    def tomatoes_orders(self, depth: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        best_bid, bid_vol, best_ask, ask_vol = self.best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        fair = 0.5 * (best_bid + best_ask)

        if best_ask < fair:
            qty = self.clip_buy(min(self.T_TAKE_SIZE, ask_vol), position, self.T_POSITION_LIMIT)
            if qty > 0:
                orders.append(Order("TOMATOES", best_ask, qty))
                position += qty

        if best_bid > fair:
            qty = self.clip_sell(min(self.T_TAKE_SIZE, bid_vol), position, self.T_POSITION_LIMIT)
            if qty > 0:
                orders.append(Order("TOMATOES", best_bid, -qty))
                position -= qty

        fair_buy = int(fair // 1)
        fair_sell = int(-(-fair // 1))
        if position <= -self.T_FLATTEN_THRESHOLD:
            qty = self.clip_buy(min(-position, self.T_TAKE_SIZE), position, self.T_POSITION_LIMIT)
            if qty > 0:
                orders.append(Order("TOMATOES", fair_buy, qty))
                position += qty
        elif position >= self.T_FLATTEN_THRESHOLD:
            qty = self.clip_sell(min(position, self.T_TAKE_SIZE), position, self.T_POSITION_LIMIT)
            if qty > 0:
                orders.append(Order("TOMATOES", fair_sell, -qty))
                position -= qty

        bid_quote = min(best_bid + 1, fair_buy)
        ask_quote = max(best_ask - 1, fair_sell)
        if bid_quote >= fair:
            bid_quote = fair_buy - 1
        if ask_quote <= fair:
            ask_quote = fair_sell + 1

        buy_qty = self.clip_buy(self.T_BASE_SIZE, position, self.T_POSITION_LIMIT)
        sell_qty = self.clip_sell(self.T_BASE_SIZE, position, self.T_POSITION_LIMIT)

        if buy_qty > 0:
            orders.append(Order("TOMATOES", int(bid_quote), buy_qty))
        if sell_qty > 0:
            orders.append(Order("TOMATOES", int(ask_quote), -sell_qty))

        return orders

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        if "EMERALDS" in state.order_depths:
            result["EMERALDS"] = self.emeralds_orders(
                state.order_depths["EMERALDS"],
                state.position.get("EMERALDS", 0),
            )

        if "TOMATOES" in state.order_depths:
            result["TOMATOES"] = self.tomatoes_orders(
                state.order_depths["TOMATOES"],
                state.position.get("TOMATOES", 0),
            )

        return result, 0, ""
