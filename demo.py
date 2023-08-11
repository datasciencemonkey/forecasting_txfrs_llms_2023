# %%
import sys
from rich import print
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import plotly.express as px
print(sys.version)
# Data comes from kaggle. copy in the repo.
# https://www.kaggle.com/datasets/ujjwalchowdhury/walmartcleaned

# %%
import numpy as np
import pandas as pd
from IPython.display import display, Markdown

import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, PatchTST

# from neuralforecast.utils import AirPassengersDF

# %%
df = pd.read_csv("forecasting_dept_data.csv")
# %%
print(f"info: all depts have week starts of {df.day_of_week.unique()[0]}")
print("info: converting wk_date to datetime")
df["wk_date"] = pd.to_datetime(df["wk_date"])

# %%
# Get 1 dept data and plot it
one_dept_df = df[df["dept"] == df.dept.iloc[0]]
(one_dept_df.sort_values("wk_date").plot(kind="line", x="wk_date", y="total_sales"))
print(f"info: dept {one_dept_df.dept.iloc[0]} has {len(one_dept_df)} weeks of data")
# %%
one_dept_df = one_dept_df.sort_values("wk_date")
train_cut_off_date = one_dept_df.iloc[round(len(one_dept_df) * 0.8)]["wk_date"]
print(f"info: train cut off date is {train_cut_off_date}")
train_df = df[df["wk_date"] <= train_cut_off_date][["dept", "wk_date", "total_sales"]]
test_df = df[df["wk_date"] > train_cut_off_date][["dept", "wk_date", "total_sales"]]

# %%
train_df[train_df.dept == one_dept_df.dept.unique()[0]].plot(
    kind="line", x="wk_date", y="total_sales"
)
test_df[test_df.dept == one_dept_df.dept.unique()[0]].plot(
    kind="line", x="wk_date", y="total_sales"
)
# %%
horizon = len(test_df[test_df.dept == one_dept_df.dept.unique()[0]])
models = [
    NBEATS(input_size=3 * horizon, h=horizon, max_steps=500),
    NHITS(input_size=3 * horizon, h=horizon, max_steps=500),
    PatchTST(input_size=3 * horizon, h=horizon, max_steps=500),
]
nf = NeuralForecast(models=models, freq="W-FRI")
# %%
import math

train_df["total_sales"] = train_df["total_sales"].apply(lambda x: math.log(x))
# %%
import time

start_time = time.perf_counter()
y_train_df = train_df[["dept", "wk_date", "total_sales"]]
y_train_df.columns = ["unique_id", "ds", "y"]
nf.fit(y_train_df, verbose=True)
end_time = time.perf_counter()
print(f"info: training took {end_time - start_time:0.4f} seconds for {len(models)} models")
# %%
Y_hat_df = nf.predict().reset_index()

# %%
Y_test_df = test_df[["dept", "wk_date", "total_sales"]]
Y_test_df["total_sales"] = Y_test_df["total_sales"].apply(lambda x: math.log(x))
Y_test_df.columns = ["unique_id", "ds", "y"]
# %%
one_dept_test_df = Y_test_df[Y_test_df.unique_id == one_dept_df.dept.unique()[0]]
one_dept_test_df = one_dept_test_df.sort_values(by="ds")
Y_hat_one_dept_test_df = Y_hat_df[Y_hat_df.unique_id == one_dept_df.dept.unique()[0]]
Y_hat_one_dept_test_df = Y_hat_one_dept_test_df.sort_values(by="ds")


# %%
import matplotlib.pyplot as plt

# Plot
fig, ax = plt.subplots(figsize=(20, 7))
Y_hat_one_dept_test_df.plot(kind="line", x="ds", y="NBEATS", ax=ax, label="NBEATS")
Y_hat_one_dept_test_df.plot(kind="line", x="ds", y="NHITS", ax=ax, label="NHITS")
Y_hat_one_dept_test_df.plot(kind="line", x="ds", y="PatchTST", ax=ax, label="PatchTST")
one_dept_test_df.plot(kind="line", x="ds", y="y", ax=ax, label="Actual")
# Set the chart title and axis labels
ax.set_title("Weekly Sales Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Weekly Sales")
ax.legend()
plt.show()
# %%
# Now to calculate performance
Y_hat_df["NBEATS"] = Y_hat_df["NBEATS"].apply(lambda x: math.exp(x))
Y_hat_df["NHITS"] = Y_hat_df["NHITS"].apply(lambda x: math.exp(x))
Y_hat_df["PatchTST"] = Y_hat_df["PatchTST"].apply(lambda x: math.exp(x))


# %%
from neuralforecast.losses.pytorch import MAPE, SMAPE
from neuralforecast.losses.numpy import mape, smape


def collect_metrics(
    test_df: pd.DataFrame = test_df,
    Y_hat_df: pd.DataFrame = Y_hat_df,
    algorithm: str = "PatchTST",
):
    metrics_dict = {}
    for i in test_df.dept.unique():
        nbeats_mape = mape(
            test_df[test_df["dept"] == i].sort_values("wk_date")["total_sales"].values,
            Y_hat_df[Y_hat_df["unique_id"] == i].sort_values("ds")[algorithm].values,
        )
        nbeats_smape = smape(
            test_df[test_df["dept"] == i].sort_values("wk_date")["total_sales"].values,
            Y_hat_df[Y_hat_df["unique_id"] == i].sort_values("ds")[algorithm].values,
        )
    
        metrics_dict[i] = [nbeats_mape, nbeats_smape]

    return metrics_dict


nbeats_metrics_df = pd.DataFrame.from_dict(
    collect_metrics(algorithm="NBEATS"), orient="index", columns=["MAPE", "SMAPE"]
)
patchtst_metrics_df = pd.DataFrame.from_dict(
    collect_metrics(algorithm="PatchTST"), orient="index", columns=["MAPE", "SMAPE"]
)
nhits_metrics_df = pd.DataFrame.from_dict(
    collect_metrics(algorithm="NHITS"), orient="index", columns=["MAPE", "SMAPE"]
)

# %%
eval_df = pd.concat(
    [nbeats_metrics_df, patchtst_metrics_df, nhits_metrics_df],
    axis=1,
    keys=["NBEATS", "PatchTST", "NHITS"],
)

eval_df.plot(
    kind="bar",
    y=[("NBEATS", "SMAPE"), ("PatchTST", "SMAPE"), ("NHITS", "SMAPE")],
    title="SMAPE by department",
)

# show the plots
plt.show()
# %%
eval_df["champion"] = eval_df[
    [("NBEATS", "SMAPE"), ("PatchTST", "SMAPE"), ("NHITS", "SMAPE")]
].idxmin(axis=1)
# %%
eval_df["champion_model"] = eval_df["champion"].apply(lambda x: x[0])


### Now for completeness sake let's try ARIMA against PatchTST

start_time = time.perf_counter()
sf = StatsForecast(models=[AutoARIMA(season_length=52)], freq="W-FRI")
sf.fit(y_train_df)
end_time = time.perf_counter()
print(f"info: training took {end_time - start_time:0.4f} seconds for {len(sf.fitted_.flatten())} ARIMA models")
arima_preds = sf.predict(h=28, level=[95])

arima_preds["AutoARIMA"] = arima_preds["AutoARIMA"].apply(lambda x: math.exp(x))


def collect_metrics_arima(
    test_df: pd.DataFrame = test_df,
    Y_hat_df: pd.DataFrame = arima_preds.reset_index(),
    algorithm: str = "AutoARIMA",
):
    metrics_dict = {}
    for i in test_df.dept.unique():
        nbeats_mape = mape(
            test_df[test_df["dept"] == i].sort_values("wk_date")["total_sales"].values,
            Y_hat_df[Y_hat_df["unique_id"] == i].sort_values("ds")[algorithm].values,
        )
        nbeats_smape = smape(
            test_df[test_df["dept"] == i].sort_values("wk_date")["total_sales"].values,
            Y_hat_df[Y_hat_df["unique_id"] == i].sort_values("ds")[algorithm].values,
        )
        # format the output to show in % format
        # print(f"MAPE for departmentID-{i} is {nbeats_mape * 100 :.2f}%")
        # print(f"SMAPE for departmentID{i} is {nbeats_smape * 100 :.2f}%")
        metrics_dict[i] = [nbeats_mape, nbeats_smape]

    return metrics_dict


arima_metrics_df = pd.DataFrame.from_dict(
    collect_metrics_arima(algorithm="AutoARIMA"),
    orient="index",
    columns=["MAPE", "SMAPE"],
)

compare_df = pd.concat(
    [patchtst_metrics_df, arima_metrics_df], axis=1, keys=["PatchTST", "AutoARIMA"]
)

compare_df["champion"] = compare_df[
    [("PatchTST", "SMAPE"), ("AutoARIMA", "SMAPE")]
].idxmin(axis=1)


compare_df = compare_df.reset_index()
compare_df.columns = [
    "unique_id",
    "PatchTST_MAPE",
    "PatchTST_SMAPE",
    "AutoARIMA_MAPE",
    "AutoARIMA_SMAPE",
    "champion",
]
# compare_df.to_csv("comparison.csv", index=False)

# %%
# compile results into a single dataframe
arima_preds = arima_preds.reset_index().rename(columns={"ds": "wk_date"})[
    ["unique_id", "wk_date", "AutoARIMA"]
]
Y_hat_df = Y_hat_df.rename(columns={"ds": "wk_date"})[
    ["unique_id", "wk_date", "PatchTST"]
]


# %%
# get the results based on whether the model is the champion or not
forecasts_df = pd.DataFrame()
for i in range(len(compare_df)):
    if compare_df.iloc[i]["champion"][0] == "PatchTST":
        forecasts_df = pd.concat(
            [
                forecasts_df,
                Y_hat_df[Y_hat_df["unique_id"] == compare_df.iloc[i]["unique_id"]],
            ]
        )
    else:
        forecasts_df = pd.concat(
            [
                forecasts_df,
                arima_preds[
                    arima_preds["unique_id"] == compare_df.iloc[i]["unique_id"]
                ],
            ]
        )

# %%
# create a column such that the column `forecast` has a value from either PatchTST or AutoARIMA based on if the value is not a NaN
forecasts_df["sales"] = forecasts_df.apply(
    lambda x: x["PatchTST"] if not math.isnan(x["PatchTST"]) else x["AutoARIMA"],
    axis=1,
)
forecasts_df['ctype'] = 'forecasts'
# %%
list_departments = list(forecasts_df.unique_id.unique())
# Arbitrary department names
dept_names = [
    "Accessories",
    "Appliances",
    "Beauty",
    "Bedding",
    "Beverages",
    "Canned Foods",
    "Cereal",
    "Cleaning Supplies",
    "Dairy",
    "Dinner Foods",
]
# Attach department names to department IDs
dept_dict = dict(zip(list_departments, dept_names))
forecasts_df["dept_name"] = forecasts_df["unique_id"].map(dept_dict)

# %%
# add dept_name to train_df
train_df["dept_name"] = train_df["dept"].map(dept_dict)
#undo log transform
train_df["sales"] = train_df["total_sales"].apply(lambda x: math.exp(x))
train_df['ctype'] = 'actuals'
train_df.drop(columns=['total_sales'], inplace=True)
# %%
cols = ['dept_name', 'wk_date', 'sales', 'ctype']
all_df = pd.concat([train_df[cols], forecasts_df[cols]], axis=0)
# %%
# Using all_df plot the actuals and forecasts for Cereral department
# fig = px.line(
#     all_df[all_df["dept_name"] == "Cereal"],
#     x="wk_date",
#     y="sales",
#     color="ctype",
#     title="Actuals vs Forecasts for Cereal Department",
# )
# %%
# all_df.to_csv("actuals_and_forecasts.csv", index=False)
eval_df.head(10)

# %%
compare_df.head(10)
# %%
