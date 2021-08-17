import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def prepare_fifa_data(fifa_data: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    time_series = fifa_data[['country_full', 'rank']]

    time_series = time_series.pivot(
        columns='country_full',
        values='rank'
    )
    time_series = time_series[countries]
    return time_series


if __name__ == '__main__':
    raw_data = pd.read_csv(Path('data') / 'fifa_ranking-2021-05-27.csv',
                           index_col='rank_date',
                           parse_dates=True)

    fifa_data = prepare_fifa_data(raw_data, ['Brazil', 'Egypt', 'Albania', 'Spain', 'Netherlands', 'Israel'])
    ax = sns.lineplot(data=fifa_data)
    ax.invert_yaxis()  # show low numbers as high rank (e.g. 1st place)
    plt.show()
