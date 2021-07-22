import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import termcolor

from enum import Enum, auto


snp_data = pd.read_csv("sp-500-historical-annual-returns.csv")
pd.to_datetime(snp_data['date'], format='%Y-%m-%d')
snp_data["year"] = pd.DatetimeIndex(snp_data['date']).year


class Strategy(Enum):
    LumpSum = auto()
    DollarCostAveraging = auto()


def change_to_multiplier(change: float) -> float:
    return (change / 100) + 1


def invest_lump_sum(amount: int, year_begin: int, year_end: int):
    period_data = snp_data[snp_data.year >= year_begin]
    period_data = period_data[snp_data.year <= year_end]
    for _, row in period_data.iterrows():
        # change_raw = row.value / 100
        # change = 1 + change_raw
        amount *= change_to_multiplier(row.value)
    return amount


def invest_dollar_cost_averaging(yearly_amount: int, year_begin: int, year_end: int):
    period_data = snp_data[snp_data.year >= year_begin]
    period_data = period_data[snp_data.year <= year_end]
    amount = 0
    for _, row in period_data.iterrows():
        amount += yearly_amount
        # print(f"year: {row.year}, amount {amount}, amount at year end: {amount * change_to_multiplier(row.value)}")
        amount *= change_to_multiplier(row.value)
    return amount


def passive_investment_simulator(period: int,
                                 strategy: Strategy,
                                 begin_year=1960,
                                 end_year=2050,
                                 ):
    investment_period = period
    bad_years = int()
    great_years = int()
    all_returns = []
    for begin in range(begin_year, end_year, 1):
        beginning_sum = 1000000
        end = begin + investment_period - 1
        if end > snp_data.year.max():
            break

        invest = invest_lump_sum if strategy is strategy.LumpSum else invest_dollar_cost_averaging
        fund = invest(beginning_sum, begin, end)
        msg = f"investing: {beginning_sum}$ between: {begin}-{end}" \
              f" would leave you with: {fund:.2f}$"

        denominator = beginning_sum if strategy is strategy.LumpSum else beginning_sum * investment_period
        returns = fund / denominator
        all_returns.append(returns)
        if returns < 1:  # bad period, lost money
            msg = termcolor.colored(msg, color='red')
            bad_years += 1
        elif returns > 4:  # great period made 4X
            msg = termcolor.colored(msg, color='green')
            great_years += 1
        # print(msg)
    # print()
    # print(f"""overall investment period of: {investment_period} years
    # included: {great_years} great periods,
    # and: {bad_years} bad periods
    # """)
    return all_returns


if __name__ == '__main__':
    for i in range(1, 26):
        returns_for_period = passive_investment_simulator(i, Strategy.DollarCostAveraging)
        print(f"""returns for {i} years:
                  mean: {np.mean(returns_for_period)}
                  min: {np.min(returns_for_period)}
                  max: {np.max(returns_for_period)}
                  """)
