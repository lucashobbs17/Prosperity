import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


BASE_DIR = Path(r"C:\Users\lucas\OneDrive\Desktop\Coding\Prosperity")
TRADE_DIR = Path(r"C:\Users\lucas\AppData\Local\Temp")
SYMBOL = "INTARIAN_PEPPER_ROOT"
LOOKAHEAD = 10
RIDGE = 0.25


@dataclass
class Row:
    timestamp: int
    best_bid: float
    best_ask: float
    bid_vol: float
    ask_vol: float
    mid: float
    fair: float


def load_rows(day: int) -> List[Row]:
    rows: List[Row] = []
    with (BASE_DIR / f"prices_round_1_day_{day}.csv").open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for raw in reader:
            if raw["product"] != SYMBOL:
                continue
            bid = raw["bid_price_1"].strip()
            ask = raw["ask_price_1"].strip()
            if not bid or not ask:
                continue
            bid_vol = raw["bid_volume_1"].strip()
            ask_vol = raw["ask_volume_1"].strip()
            bid_value = float(bid)
            ask_value = float(ask)
            timestamp = int(raw["timestamp"])
            rows.append(
                Row(
                    timestamp=timestamp,
                    best_bid=bid_value,
                    best_ask=ask_value,
                    bid_vol=float(bid_vol or 0),
                    ask_vol=float(ask_vol or 0),
                    mid=0.5 * (bid_value + ask_value),
                    fair=(10000 if day == -2 else 11000 if day == -1 else 12000) + 0.001 * timestamp,
                )
            )
    return rows


def load_trade_signal(day: int) -> Dict[int, float]:
    signal_by_timestamp: Dict[int, float] = {}
    with (TRADE_DIR / f"trades_round_1_day_{day}.csv").open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        grouped: Dict[int, List[Tuple[float, float]]] = {}
        for raw in reader:
            if raw["symbol"] != SYMBOL:
                continue
            timestamp = int(raw["timestamp"])
            grouped.setdefault(timestamp, []).append(
                (float(raw["price"]), float(raw["quantity"]))
            )
        rows = {row.timestamp: row for row in load_rows(day)}
        for timestamp, trades in grouped.items():
            row = rows.get(timestamp)
            if row is None:
                continue
            total = 0.0
            for price, qty in trades:
                if qty < 7:
                    continue
                if price > row.mid:
                    total += qty
                elif price < row.mid:
                    total -= qty
            signal_by_timestamp[timestamp] = total
    return signal_by_timestamp


def build_examples(day: int):
    rows = load_rows(day)
    trade_signal = load_trade_signal(day)
    examples = []
    for index in range(24, len(rows) - LOOKAHEAD):
        row = rows[index]
        history = rows[: index + 1]
        fast = sum(item.mid for item in history[-8:]) / 8.0
        slow = sum(item.mid for item in history[-24:]) / 24.0
        older = sum(item.mid for item in history[-24:-8]) / 16.0
        future_mid = rows[index + LOOKAHEAD].mid
        target = future_mid - row.mid

        spread = row.best_ask - row.best_bid
        imbalance_denom = row.bid_vol + row.ask_vol
        imbalance = ((row.bid_vol - row.ask_vol) / imbalance_denom) if imbalance_denom else 0.0

        examples.append(
            {
                "x": [
                    1.0,
                    row.fair - row.mid,
                    fast - slow,
                    fast - older,
                    imbalance,
                    spread,
                    trade_signal.get(row.timestamp, 0.0),
                ],
                "y": target,
            }
        )
    return examples


def solve_linear_system(matrix: List[List[float]], vector: List[float]) -> List[float]:
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for pivot in range(size):
        best = max(range(pivot, size), key=lambda idx: abs(augmented[idx][pivot]))
        augmented[pivot], augmented[best] = augmented[best], augmented[pivot]
        pivot_value = augmented[pivot][pivot]
        if abs(pivot_value) < 1e-12:
            continue
        for column in range(pivot, size + 1):
            augmented[pivot][column] /= pivot_value
        for row_index in range(size):
            if row_index == pivot:
                continue
            factor = augmented[row_index][pivot]
            if factor == 0:
                continue
            for column in range(pivot, size + 1):
                augmented[row_index][column] -= factor * augmented[pivot][column]
    return [augmented[index][-1] for index in range(size)]


def fit_ridge(examples) -> List[float]:
    width = len(examples[0]["x"])
    xtx = [[0.0 for _ in range(width)] for _ in range(width)]
    xty = [0.0 for _ in range(width)]
    for example in examples:
        x = example["x"]
        y = example["y"]
        for i in range(width):
            xty[i] += x[i] * y
            for j in range(width):
                xtx[i][j] += x[i] * x[j]
    for i in range(width):
        xtx[i][i] += RIDGE
    return solve_linear_system(xtx, xty)


def score(coefficients: List[float], examples) -> Tuple[float, float]:
    mse = 0.0
    hit = 0
    count = 0
    for example in examples:
        pred = sum(weight * value for weight, value in zip(coefficients, example["x"]))
        actual = example["y"]
        mse += (pred - actual) ** 2
        if pred == 0 or actual == 0:
            continue
        count += 1
        if (pred > 0 and actual > 0) or (pred < 0 and actual < 0):
            hit += 1
    return mse / len(examples), (hit / count if count else 0.0)


def print_coefficients(coefficients: List[float]):
    names = [
        "intercept",
        "fair_gap",
        "fast_minus_slow",
        "fast_minus_older",
        "imbalance",
        "spread",
        "large_trade_signal",
    ]
    print("coefficients")
    for name, value in zip(names, coefficients):
        print(f"  {name}: {value:.6f}")


def main():
    day_examples = {day: build_examples(day) for day in (-2, -1, 0)}
    train = day_examples[-2] + day_examples[-1]
    test = day_examples[0]
    coefficients = fit_ridge(train)
    train_mse, train_hit = score(coefficients, train)
    test_mse, test_hit = score(coefficients, test)
    print_coefficients(coefficients)
    print(f"train mse {train_mse:.6f} hit {train_hit:.4f}")
    print(f"test mse {test_mse:.6f} hit {test_hit:.4f}")


if __name__ == "__main__":
    main()
