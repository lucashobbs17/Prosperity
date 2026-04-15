import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(r"C:\Users\lucas\OneDrive\Desktop\Coding\Prosperity")
PRODUCT = "INTARIAN_PEPPER_ROOT"
OUTPUT = BASE_DIR / "intarian_pepper_trade_size_impact.png"


def load_prices(day: int):
    path = BASE_DIR / f"prices_round_1_day_{day}.csv"
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            if row["product"] != PRODUCT:
                continue
            bid = row["bid_price_1"].strip()
            ask = row["ask_price_1"].strip()
            if not bid or not ask:
                continue
            timestamp = int(row["timestamp"])
            bid = float(bid)
            ask = float(ask)
            mid = 0.5 * (bid + ask)
            rows.append(
                {
                    "timestamp": timestamp,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                }
            )
    return rows


def load_trades(day: int):
    path = Path(r"C:\Users\lucas\AppData\Local\Temp") / f"trades_round_1_day_{day}.csv"
    trades = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            if row["symbol"] != PRODUCT:
                continue
            trades.append(
                {
                    "timestamp": int(row["timestamp"]),
                    "price": float(row["price"]),
                    "quantity": int(float(row["quantity"])),
                }
            )
    return trades


def infer_trade_side(trade_price: float, mid: float):
    if trade_price > mid:
        return 1
    if trade_price < mid:
        return -1
    return 0


def future_mid(prices, index, steps):
    future_index = min(len(prices) - 1, index + steps)
    return prices[future_index]["mid"]


def summarize_impacts(samples):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["quantity"]].append(sample["impact"])
    summary = []
    for quantity in sorted(grouped):
        impacts = grouped[quantity]
        summary.append(
            {
                "quantity": quantity,
                "avg_impact": sum(impacts) / len(impacts),
                "count": len(impacts),
            }
        )
    return summary


def build_day_samples(day: int, horizon_steps: int = 10):
    prices = load_prices(day)
    trades = load_trades(day)
    price_by_timestamp = {row["timestamp"]: (idx, row) for idx, row in enumerate(prices)}
    samples = []
    for trade in trades:
        found = price_by_timestamp.get(trade["timestamp"])
        if found is None:
            continue
        index, row = found
        side = infer_trade_side(trade["price"], row["mid"])
        if side == 0:
            continue
        fwd_mid = future_mid(prices, index, horizon_steps)
        move = fwd_mid - row["mid"]
        impact = side * move
        samples.append(
            {
                "timestamp": trade["timestamp"],
                "mid": row["mid"],
                "quantity": trade["quantity"],
                "side": side,
                "impact": impact,
            }
        )
    return prices, samples, summarize_impacts(samples)


def main():
    days = [-2, -1, 0]
    fig, axes = plt.subplots(len(days), 2, figsize=(14, 12), constrained_layout=True)

    for row_index, day in enumerate(days):
        prices, samples, summary = build_day_samples(day)

        ax_price = axes[row_index, 0]
        ax_scatter = axes[row_index, 1]

        timestamps = [row["timestamp"] for row in prices]
        mids = [row["mid"] for row in prices]
        ax_price.plot(timestamps, mids, color="black", linewidth=1.3)

        buy_samples = [sample for sample in samples if sample["side"] > 0]
        sell_samples = [sample for sample in samples if sample["side"] < 0]
        if buy_samples:
            ax_price.scatter(
                [sample["timestamp"] for sample in buy_samples],
                [sample["mid"] for sample in buy_samples],
                s=[sample["quantity"] * 14 for sample in buy_samples],
                color="#1f77b4",
                alpha=0.65,
                label="buy-like trade",
            )
        if sell_samples:
            ax_price.scatter(
                [sample["timestamp"] for sample in sell_samples],
                [sample["mid"] for sample in sell_samples],
                s=[sample["quantity"] * 14 for sample in sell_samples],
                color="#d62728",
                alpha=0.65,
                label="sell-like trade",
            )

        ax_price.set_title(f"Day {day}: price with trade markers")
        ax_price.set_ylabel("Price")
        ax_price.grid(alpha=0.2)
        if row_index == 0:
            ax_price.legend(loc="upper left")

        if samples:
            colors = ["#1f77b4" if sample["side"] > 0 else "#d62728" for sample in samples]
            ax_scatter.scatter(
                [sample["quantity"] for sample in samples],
                [sample["impact"] for sample in samples],
                color=colors,
                alpha=0.65,
                s=35,
            )
            ax_scatter.axhline(0, color="black", linewidth=1)

            if summary:
                ax_scatter.plot(
                    [item["quantity"] for item in summary],
                    [item["avg_impact"] for item in summary],
                    color="#2ca02c",
                    linewidth=2,
                    marker="o",
                    label="avg by size",
                )

        ax_scatter.set_title(f"Day {day}: size vs 10-step signed move")
        ax_scatter.set_xlabel("Trade quantity")
        ax_scatter.set_ylabel("Signed future move")
        ax_scatter.grid(alpha=0.2)
        if row_index == 0:
            ax_scatter.legend(loc="upper left")

    fig.suptitle("INTARIAN_PEPPER_ROOT trade size vs subsequent move", fontsize=16)
    fig.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()
