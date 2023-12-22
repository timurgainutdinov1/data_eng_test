import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", 20, "display.max_columns", 60)


def read_file(file_name):
    return pd.read_csv(file_name)


def read_types(file_name):
    dtypes = {}
    with open(file_name, "r") as file:
        dtypes = json.load(file)

    for key in dtypes.keys():
        if dtypes[key] == 'category':
            dtypes[key] = pd.CategoricalDtype
        else:
            dtypes[key] = np.dtype(dtypes[key])

    return dtypes


need_dtypes = read_types("dtypes.json")

dataset = pd.read_csv("df.csv",
                      usecols=lambda x: x in need_dtypes.keys(),
                      dtype=need_dtypes)

print(dataset.info(memory_usage='deep'))
print(dataset.isnull().sum())

# dataset['attendance'].fillna(dataset['attendance'].mean(), inplace=True)
dataset['length_minutes'].fillna(dataset['length_minutes'].mean(), inplace=True)
dataset['v_hits'].fillna(dataset['v_hits'].mean(), inplace=True)

# # Распределение посещаемости
# plt.figure()
# plt.hist(dataset['attendance'], color='blue', edgecolor='black', bins=int(10000/1000))
# plt.title('Распределение посещаемости')
# plt.xlabel('Посещаемость')
# plt.ylabel('Частота')
# plt.savefig('graph/game_logs/graph1.png', dpi=300)
#
# # Средняя посещаемость по дням недели
# plt.figure()
# mean_attendance = dataset.groupby('day_of_week', observed=False)['attendance'].mean()
# plt.bar(mean_attendance.index, mean_attendance.values, zorder=2)
# plt.grid(True)
# plt.title('Средняя посещаемость по дням недели')
# plt.xlabel('День недели')
# plt.ylabel('Посещаемость')
# plt.savefig('graph/game_logs/graph2.png', dpi=300)

# # Диаграмма рассеяния: Посещаемость и количество ударов v_hits
# plt.figure()
# plt.scatter(dataset['attendance'], dataset['v_hits'], alpha=0.5)
# plt.xlabel('Посещаемость')
# plt.ylabel('Количество ударов v_hits')
# plt.title('Диаграмма рассеяния: Посещаемость и количество ударов v_hits')
# plt.savefig('graph/game_logs/graph3.png', dpi=300)

# Соотношение очков v_score и h_score
# plt.figure()
# score_labels = ['v_score', 'h_score']
# score_values = [dataset['v_score'].sum(), dataset['h_score'].sum()]
# plt.pie(score_values, labels=score_labels, autopct='%1.1f%%')
# plt.title('Соотношение очков v_score и h_score')
# plt.savefig('graph/game_logs/graph4.png', dpi=300)

# Cуммарные посещаемости за год
# dataset['date'] = pd.to_datetime(dataset['date'], format='%Y%m%d')
# dataset['year'] = dataset['date'].dt.year
# grouped_df = dataset.groupby('year')['attendance'].sum().reset_index()
# plt.figure()
# plt.plot(grouped_df['year'], grouped_df['attendance'], zorder=2)
# plt.grid(True)
# plt.xlabel('Год')
# plt.ylabel('Суммарная посещаемость')
# plt.title('Cуммарные посещаемости за год')
# plt.savefig('graph/game_logs/graph5.png', dpi=300)
