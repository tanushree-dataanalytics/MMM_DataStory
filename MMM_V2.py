import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_absolute_percentage_error)
from sklearn.model_selection import TimeSeriesSplit

# LOAD DATA
df = pd.read_csv("C:/Users/kaush/simulated_marketing_data.csv")
df["DATE"] = pd.to_datetime(df["DATE"])
df = (
    df.sort_values("DATE")
    .reset_index(drop=True)
    )
print(f"Dataset: {len(df)} observations | " f"{df['DATE'].min().date()} " f"to " f"{df['DATE'].max().date()}")

# TRAIN - TEST SPLIT
split = int(len(df) * 0.80)

train_df = df.iloc[:split].copy()
test_df = df.iloc[split:].copy()

print("\nTRAIN / TEST SPLIT")
print(f"Train observations: {len(train_df)}")
print(f"Test observations : {len(test_df)}")

# COMPETITOR SALES DIAGNOSTIC
print("\nCOMPETITOR SALES DIAGNOSTIC")

if "competitor_sales" in df.columns:
    competitor_corr = (
        df["competitor_sales"]
        .corr(df["revenue"])
    )

    print(
        f"Correlation with revenue: "
        f"{competitor_corr:.4f}"
    )

    if abs(competitor_corr) > 0.98:
        print(
            "WARNING: Near-perfect correlation detected."
        )
        print(
            "Potential leakage or duplicated information."
        )

    elif abs(competitor_corr) > 0.90:
        print(
            "CAUTION: Extremely strong relationship."
        )

    else:
        print(
            "No obvious leakage signal."
        )

    print("\nSummary Statistics")
    print(
        df[
            ["revenue", "competitor_sales"]
        ].describe()
    )

else:
    print(
        "competitor_sales not found."
    )

# ADSTOCK FUNCTION
def adstock(series, decay):
        result = np.zeros(len(series))

        result[0] = series[0]

        for i in range(1, len(series)):
            result[i] = (
                series[i]
                + decay * result[i - 1]
            )
            
        return result

# TRAIN ADSTOCK
train_df["tv_adstock"] = adstock(
train_df["tv_S"].values,
0.6
)

train_df["radio_adstock"] = adstock(
train_df["radio_S"].values,
0.3
)

train_df["search_adstock"] = adstock(
train_df["paid_search_S"].values,
0.1
)

# TEST ADSTOCK
test_df["tv_adstock"] = adstock(
test_df["tv_S"].values,
0.6
)

test_df["radio_adstock"] = adstock(
test_df["radio_S"].values,
0.3
)

test_df["search_adstock"] = adstock(
test_df["paid_search_S"].values,
0.1
)

# SATURATION FUNCTION
def saturation_mm_transform(series, K):
    return series / (series + K)

# FIT SATURATION PARAMETERS USING TRAIN ONLY
tv_K = np.median(
train_df.loc[
train_df["tv_adstock"] > 0,
"tv_adstock"
]
)

radio_K = np.median(
train_df.loc[
train_df["radio_adstock"] > 0,
"radio_adstock"
]
)

search_K = np.median(
train_df.loc[
train_df["search_adstock"] > 0,
"search_adstock"
]
)

print("\nSATURATION PARAMETERS")
print(f"TV K     : {tv_K:.2f}")
print(f"Radio K  : {radio_K:.2f}")
print(f"Search K : {search_K:.2f}")

# APPLY SATURATION TO TRAIN
train_df["tv_sat"] = saturation_mm_transform(
train_df["tv_adstock"],
tv_K
)

train_df["radio_sat"] = saturation_mm_transform(
train_df["radio_adstock"],
radio_K
)

train_df["search_sat"] = saturation_mm_transform(
train_df["search_adstock"],
search_K
)

# APPLY SAME PARAMETERS TO TEST
test_df["tv_sat"] = saturation_mm_transform(
test_df["tv_adstock"],
tv_K
)

test_df["radio_sat"] = saturation_mm_transform(
test_df["radio_adstock"],
radio_K
)

test_df["search_sat"] = saturation_mm_transform(
test_df["search_adstock"],
search_K
)

# SEASONALITY + TREND
for dataset in [train_df, test_df]:
    dataset["month"] = (
    dataset["DATE"].dt.month
)

train_months = pd.get_dummies(
    train_df["month"],
    prefix="month",
    drop_first=True
).astype(int)

test_months = pd.get_dummies(
    test_df["month"],
    prefix="month",
    drop_first=True
).astype(int)

# Align columns
train_months, test_months = train_months.align(
    test_months,
    join="outer",
    axis=1,
    fill_value=0
)

train_df = pd.concat(
[train_df, train_months],
axis=1
)

test_df = pd.concat(
[test_df, test_months],
axis=1
)

month_cols = list(train_months.columns)
train_df["trend"] = np.arange(
len(train_df)
)

test_df["trend"] = np.arange(
len(test_df)
)

# FEATURES
MEDIA = [
"tv_sat",
"radio_sat",
"search_sat"
]

feature_cols = (
MEDIA
+ ["trend"]
+ month_cols
)

# OLS
X_train = sm.add_constant(
train_df[feature_cols]
)

X_test = sm.add_constant(
test_df[feature_cols]
)

y_train = train_df["revenue"]
y_test = test_df["revenue"]

ols = sm.OLS(
y_train,
X_train
).fit()

pred_ols = ols.predict(X_test)

ols_mape = (
mean_absolute_percentage_error(
y_test,
pred_ols
) * 100
)

print("\nOLS RESULTS")
print(
f"OOS MAPE: {ols_mape:.2f}%"
)

# TIME SERIES CROSS VALIDATION
tscv = TimeSeriesSplit(
    n_splits=5
)

print("\nTIME SERIES CROSS VALIDATION")
print(
    f"Number of splits: "
    f"{tscv.n_splits}"
)

# RIDGE
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
train_df[feature_cols]
)

X_test_scaled = scaler.transform(
test_df[feature_cols]
)

ridge = RidgeCV(
    alphas=[
        0.01,
        0.1,
        1,
        10,
        50,
        100,
        500
    ],
    cv=tscv
)

ridge.fit(
X_train_scaled,
y_train
)

pred_ridge = ridge.predict(
X_test_scaled
)

ridge_mape = (
mean_absolute_percentage_error(
y_test,
pred_ridge
) * 100
)

print("\nRIDGE RESULTS")
print(
f"OOS MAPE : {ridge_mape:.2f}%"
)
print(
f"Best Alpha: {ridge.alpha_}"
)

print(
    "Validation method: "
    "TimeSeriesSplit"
)

print("\nMMM V3 COMPLETE")

print("Phase 1: Data leakage removed.")
print("Phase 2: TimeSeriesSplit implemented.")
print("Phase 3: Competitor diagnostics completed.")

print(df[['tv_S','radio_S','paid_search_S']].corr())
print(train_df[['tv_sat','radio_sat','search_sat']].corr())