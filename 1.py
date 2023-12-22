import json
import pandas as pd
import os

pd.set_option("display.max_rows", 20, "display.max_columns", 60)


def read_file(file_name):
    return pd.read_csv(file_name)
    # df = pd.read_csv(datasets[year], chunksize=chunksize, compression='gzip'])


def read_file_and_get_memory_stat_by_column(file_path_name):
    df = read_file(file_path_name)
    file_size = os.path.getsize(file_path_name)
    print(f"file size           = {file_size // 1024:10} КБ")
    memory_usage_stat = df.memory_usage(deep=True)
    total_memory_usage = memory_usage_stat.sum()
    print(f"file in memory size = {total_memory_usage // 1024:10} КБ")
    column_stat = list()
    for key in df.dtypes.keys():
        column_stat.append({
            "column_name": key,
            "memory_abs": memory_usage_stat[key] // 1024,
            "memory_per": round(memory_usage_stat[key] / total_memory_usage * 100, 4),
            "dtype": df.dtypes[key]
        })
    column_stat.sort(key=lambda x: x['memory_abs'], reverse=True)
    for column in column_stat:
        print(f"{column['column_name']:30}: {column['memory_abs']:10} КБ:"
              f" {column['memory_per']:10}%: {column['dtype']}")

    return df


def mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2
    return "{:03.2f} МБ".format(usage_mb)


def opt_obj(df):
    converted_obj = pd.DataFrame()
    dataset_obj = df.select_dtypes(include=['object']).copy()

    for col in dataset_obj.columns:
        num_unique_values = len(dataset_obj[col].unique())
        num_total_values = len(dataset_obj[col])
        if (num_unique_values / num_total_values) < 0.5:
            converted_obj.loc[:, col] = dataset_obj[col].astype('category')
        else:
            converted_obj.loc[:, col] = dataset_obj[col]

    print(mem_usage(dataset_obj))
    print(mem_usage(converted_obj))

    compare_objs = pd.concat([dataset_obj.dtypes, converted_obj.dtypes], axis=1)
    compare_objs.columns = ['before', 'after']
    print(compare_objs)

    return converted_obj


def opt_int(df):
    dataset_int = df.select_dtypes(include=['int'])
    """
    downcast:
            - 'integer' or 'signed': smallest signed int dtype (min.: np.int8) 
            - 'unsigned': smallest unsigned int dtype (min.: np.uint8) 
            - 'float': smallest float dtype (min.: np.float32)
    """
    converted_int = dataset_int.apply(pd.to_numeric, downcast='unsigned')
    print(mem_usage(dataset_int))
    print(mem_usage(converted_int))

    compare_ints = pd.concat([dataset_int.dtypes, converted_int.dtypes], axis=1)
    compare_ints.columns = ['before', 'after']
    print(compare_ints)

    return converted_int


def opt_float(df):
    dataset_float = df.select_dtypes(include=['float'])
    converted_float = dataset_float.apply(pd.to_numeric, downcast='float')

    print(mem_usage(dataset_float))
    print(mem_usage(converted_float))

    compare_floats = pd.concat([dataset_float.dtypes, converted_float.dtypes], axis=1)
    compare_floats.columns = ['before', 'after']
    print(compare_floats)

    return converted_float


file_path_name = 'data/[1]game_logs.csv'

dataset = read_file_and_get_memory_stat_by_column(file_path_name)

print(dataset.columns)

optimized_dataset = dataset.copy()

converted_obj = opt_obj(dataset)
converted_int = opt_int(dataset)
converted_float = opt_float(dataset)

optimized_dataset[converted_obj.columns] = converted_obj
optimized_dataset[converted_int.columns] = converted_int
optimized_dataset[converted_float.columns] = converted_float

print(mem_usage(dataset))
print(mem_usage(optimized_dataset))

need_column = dict()
column_names = ['date', 'day_of_week', 'v_score',
                'h_score', 'park_id', 'length_minutes',
                'h_game_number', 'v_game_number', 'attendance', 'v_hits']
opt_dtypes = optimized_dataset.dtypes
for key in column_names:
    need_column[key] = opt_dtypes[key]
    print(f"{key}: {opt_dtypes[key]}")

with open("dtypes.json", 'w') as file:
    dtype_json = need_column.copy()
    for key in dtype_json.keys():
        dtype_json[key] = str(dtype_json[key])

    json.dump(dtype_json, file)

has_header = True
for chunk in pd.read_csv(file_path_name,
                         usecols=lambda x: x in column_names,
                         dtype=need_column,
                         chunksize=100_000):
    print(mem_usage(chunk))
    chunk.to_csv("df.csv", mode='a', header=has_header)
    has_header = False
