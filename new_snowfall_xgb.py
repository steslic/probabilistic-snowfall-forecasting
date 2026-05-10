import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Two-stage snowfall forecasting model (Conditional regression model) 
# Stage 1: Classification (XGBoost classifier). Output: probability of snow
# Stage 2: Regression (How much snow if it snows)? XGBoost regressor. Output: snowfall amount (inches)
# Using temperature precipitation, snow depth, lag + rolling features
# lag features (1, 3, 7 days)
# rolling averages
# seasonality 

# ---------------------------------
# 1. Load and clean data
# ---------------------------------

# SNOTEL file has metadata/header text before the actual CSV header.
# Find the line that starts with "Date" so we can skip the description block.
file_path = "grand_targhee_snotel.csv"

header_row = None
with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if line.startswith("Date,"):
            header_row = i
            break

if header_row is None:
    raise ValueError("Could not find CSV header row starting with 'Date,'")

df = pd.read_csv(file_path, skiprows=header_row, low_memory=False)

# Rename SNOTEL columns to match our pipeline
df.columns = [
    "DATE",
    "SWE",    # Snow Water Equivalent (in)
    "PREC",   # Precipitation Accumulation (in) - cumulative over water year
    "TAVG",   # degF
    "TMAX",   # degF
    "TMIN",   # degF
    "SNWD",   # Snow Depth (in)
]

# Keep only core columns
df = df[["DATE", "SWE", "PREC", "SNWD", "TAVG", "TMAX", "TMIN"]].copy()

# Parse and sort dates
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

# Convert numeric columns
for col in ["SWE", "PREC", "SNWD", "TAVG", "TMAX", "TMIN"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing values
df[["SWE", "PREC", "SNWD"]] = df[["SWE", "PREC", "SNWD"]].ffill()
df[["TAVG", "TMAX", "TMIN"]] = df[["TAVG", "TMAX", "TMIN"]].ffill()

df = df.dropna().reset_index(drop=True)

# SNOTEL units:
# SWE, PREC, SNWD are in inches
# TAVG, TMAX, TMIN are in degrees F
# Keep snow variables in inches since we want snowfall output in inches
# Convert temperature to C for freezing logic / snow ratio logic
df["TAVG"] = (df["TAVG"] - 32) * (5.0 / 9.0)
df["TMAX"] = (df["TMAX"] - 32) * (5.0 / 9.0)
df["TMIN"] = (df["TMIN"] - 32) * (5.0 / 9.0)

# Convert cumulative precipitation accumulation into daily precipitation.
# SNOTEL PREC resets each water year, so negative diffs are reset to 0.
df["PRCP"] = df["PREC"].diff()
df["PRCP"] = df["PRCP"].fillna(0)
df.loc[df["PRCP"] < 0, "PRCP"] = 0

# Use daily positive increase in SWE as snowfall proxy.
# Negative changes represent melt/settling, so clip them to 0.
df["SWE_diff_in"] = df["SWE"].diff()
df["SWE_diff_in"] = df["SWE_diff_in"].fillna(0)
df.loc[df["SWE_diff_in"] < 0, "SWE_diff_in"] = 0

# Estimate snowfall depth from SWE increase using a temperature-based snow ratio.
# Colder temperatures -> fluffier snow -> higher snowfall/SWE ratio.
def snow_ratio(temp_c):
    if temp_c <= -8:
        return 15.0
    elif temp_c <= -4:
        return 12.0
    elif temp_c <= 0:
        return 10.0
    else:
        return 8.0

df["snow_ratio"] = df["TAVG"].apply(snow_ratio)
df["SNOW"] = df["SWE_diff_in"] * df["snow_ratio"]

# Smooth snowfall to reduce noise from SWE + ratio conversion
df["SNOW"] = df["SNOW"].rolling(2).mean()

df = df.dropna().reset_index(drop=True)

# ---------------------------------
# 2. Feature engineering
# ---------------------------------
# Temperature features
df["TEMP_AVG"] = (df["TMAX"] + df["TMIN"]) / 2.0
df["TEMP_RANGE"] = df["TMAX"] - df["TMIN"]
# extra weather interaction features 
df["FREEZING_FLAG"] = (df["TEMP_AVG"] <= 0).astype(int) # above or below freezing
df["PRCP_COLD_INTERACTION"] = df["PRCP"] * df["FREEZING_FLAG"]
df["SNWD_PRCP_INTERACTION"] = df["SNWD"] * df["PRCP"] # snow depth * precipitation 

# Lag features (exact value on a past day)
for lag in [1, 2, 3, 7, 14]:
    df[f"snow_lag{lag}"] = df["SNOW"].shift(lag)
    df[f"prcp_lag{lag}"] = df["PRCP"].shift(lag)
    df[f"snwd_lag{lag}"] = df["SNWD"].shift(lag)
    df[f"tavg_lag{lag}"] = df["TEMP_AVG"].shift(lag)

# Rolling features using past data only (trends over multiple days)
for window in [3, 7, 14]:
    df[f"snow_roll_mean_{window}"] = df["SNOW"].shift(1).rolling(window).mean()
    df[f"snow_roll_max_{window}"] = df["SNOW"].shift(1).rolling(window).max()

    df[f"prcp_roll_mean_{window}"] = df["PRCP"].shift(1).rolling(window).mean()
    df[f"prcp_roll_sum_{window}"] = df["PRCP"].shift(1).rolling(window).sum()

    df[f"tavg_roll_mean_{window}"] = df["TEMP_AVG"].shift(1).rolling(window).mean()
    df[f"snwd_roll_mean_{window}"] = df["SNWD"].shift(1).rolling(window).mean()

df["snow_roll_sum_7"] = df["SNOW"].shift(1).rolling(7).sum()
df["snow_roll_sum_14"] = df["SNOW"].shift(1).rolling(14).sum()
df["prcp_roll_max_7"] = df["PRCP"].shift(1).rolling(7).max()
df["snwd_change_1"] = df["SNWD"] - df["SNWD"].shift(1)

# Seasonal features
df["month"] = df["DATE"].dt.month
df["day_of_year"] = df["DATE"].dt.dayofyear

# Cyclical encoding for seasonality
# Convert months into circular coordinates
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

# ---------------------------------
# 3. Targets
# ---------------------------------
# Predict next day's snowfall
df["target_snow"] = df["SNOW"].shift(-1)

# Snow/no snow classifier target
df["target_snow_flag"] = (df["target_snow"] > 2).astype(int)

# Drop rows created by shifting, rolling
df = df.dropna().reset_index(drop=True)

# ---------------------------------
# 4. Features
# ---------------------------------
feature_cols = [
    "SNWD", "PRCP", "TMAX", "TMIN", "TEMP_AVG", "TEMP_RANGE",
    "FREEZING_FLAG", "PRCP_COLD_INTERACTION", "SNWD_PRCP_INTERACTION", # new 
    "month", "day_of_year", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "snow_lag1", "snow_lag2", "snow_lag3", "snow_lag7", "snow_lag14",
    "prcp_lag1", "prcp_lag2", "prcp_lag3", "prcp_lag7", "prcp_lag14",
    "snwd_lag1", "snwd_lag2", "snwd_lag3", "snwd_lag7", "snwd_lag14",
    "tavg_lag1", "tavg_lag2", "tavg_lag3", "tavg_lag7", "tavg_lag14",
    "snow_roll_mean_3", "snow_roll_mean_7", "snow_roll_mean_14",
    "snow_roll_max_3", "snow_roll_max_7", "snow_roll_max_14",
    "prcp_roll_mean_3", "prcp_roll_mean_7", "prcp_roll_mean_14",
    "prcp_roll_sum_3", "prcp_roll_sum_7", "prcp_roll_sum_14",
    "tavg_roll_mean_3", "tavg_roll_mean_7", "tavg_roll_mean_14",
    "snwd_roll_mean_3", "snwd_roll_mean_7", "snwd_roll_mean_14",
    "snow_roll_sum_7", "snow_roll_sum_14", "prcp_roll_max_7", "snwd_change_1", # new 
]

X = df[feature_cols] # input features the model uses to make a prediction 
y_clf = df["target_snow_flag"] # classification target 
y_reg = df["target_snow"] # regression target (only on snow days)

# ---------------------------------
# 5. Time-based split for time series
# ---------------------------------
# training data 80%, test data 20%
split_idx = int(len(df) * 0.8) 

# split input features 
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

y_clf_train = y_clf.iloc[:split_idx]
y_clf_test = y_clf.iloc[split_idx:]

y_reg_train = y_reg.iloc[:split_idx]
y_reg_test = y_reg.iloc[split_idx:]

dates_test = df["DATE"].iloc[split_idx:] # keep test dates for plotting

print("Train rows:", len(X_train))
print("Test rows:", len(X_test))
print("Actual snow days in test set:", int(y_clf_test.sum()))

# ---------------------------------
# 6. Classifier: snow vs. no snow
# ---------------------------------
num_pos = int(y_clf_train.sum()) # number of snow days
num_neg = len(y_clf_train) - num_pos # number of no snow days
scale_pos_weight = num_neg / max(num_pos, 1)

clf = XGBClassifier(
    n_estimators=300, # number of trees
    max_depth=4, 
    learning_rate=0.03, # smaller is smoother
    subsample=0.8, 
    colsample_bytree=0.8, 
    objective="binary:logistic", # binary classification
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    # scale_pos_weight=8,
)

# train classifier
clf.fit(X_train, y_clf_train) 

# Classifier outputs
y_clf_prob = clf.predict_proba(X_test)[:, 1]
threshold = 0.72
y_clf_pred = (y_clf_prob > threshold).astype(int)

print("\nClassifier metrics:")
print("Accuracy :", round(accuracy_score(y_clf_test, y_clf_pred), 4))
print("Precision:", round(precision_score(y_clf_test, y_clf_pred, zero_division=0), 4))
print("Recall   :", round(recall_score(y_clf_test, y_clf_pred, zero_division=0), 4))
print("F1       :", round(f1_score(y_clf_test, y_clf_pred, zero_division=0), 4)) # precision + recall
print("Predicted snow days in test set:", int(y_clf_pred.sum()))

# ---------------------------------
# 7. Regressor: amount if snow occurs
# ---------------------------------
# Only train on meaningful snow days to reduce noise (>2 inches)
snow_mask = y_reg_train > 2

X_train_reg = X_train[snow_mask]
y_train_reg = y_reg_train[snow_mask]

# Log-transform target to stabilize large spikes
y_train_reg_log = np.log1p(y_train_reg)

reg = XGBRegressor(
    n_estimators=1500, 
    max_depth=6, 
    learning_rate=0.03,
    subsample=0.8, 
    colsample_bytree=0.8, 
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
)

# reg.fit(X_train_reg, y_train_reg_log)
sample_weight = np.where(y_train_reg > 8, 3, 1)
reg.fit(X_train_reg, y_train_reg_log, sample_weight=sample_weight)
# reg.fit(X_train_reg, y_train_reg)

# ---------------------------------
# 8. Combine classifier + regressor
# ---------------------------------
y_pred_final = np.zeros(len(X_test), dtype=float) # assume no snowfall initially

snow_indices = np.where(y_clf_pred == 1)[0] # indices where classifier predicts snow 
if len(snow_indices) > 0:
    reg_preds_log = reg.predict(X_test.iloc[snow_indices])
    reg_preds = np.expm1(reg_preds_log)
    # reg_preds = reg.predict(X_test.iloc[snow_indices])
    y_pred_final[snow_indices] = reg_preds

# Small-value clipping
y_pred_final[y_pred_final < 0.1] = 0.0 # final prediction

# ---------------------------------
# 9. Evaluation
# ---------------------------------
rmse = mean_squared_error(y_reg_test, y_pred_final) ** 0.5
mae = mean_absolute_error(y_reg_test, y_pred_final)

print("\nFinal model performance:")
print("RMSE:", round(rmse, 4))
print("MAE :", round(mae, 4))

# ---------------------------------
# 10. Feature importance
# ---------------------------------
clf_importance = (
    pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_
    })
    .sort_values("importance", ascending=False) # sort by importance
    .head(15)
)

reg_importance = (
    pd.DataFrame({
        "feature": feature_cols,
        "importance": reg.feature_importances_
    })
    .sort_values("importance", ascending=False)
    .head(15)
)

print("\nTop classifier features:")
print(clf_importance.to_string(index=False))

print("\nTop regressor features:")
print(reg_importance.to_string(index=False))

# ---------------------------------
# 11. Save predictions
# ---------------------------------
results = pd.DataFrame({
    "DATE": dates_test.values,
    "Actual_SNOW_Next_Day": y_reg_test.values,
    "Predicted_SNOW_Next_Day": y_pred_final,
    "Snow_Probability": y_clf_prob,
    "Predicted_Snow_Flag": y_clf_pred,
})

results.to_csv("snowfall_predictions_xgb.csv", index=False)
print("\nSaved predictions to snowfall_predictions_xgb.csv")

# ---------------------------------
# 12. Exploratory data analysis
# ---------------------------------
# Snowfall distribution histogram
plt.figure(figsize=(8,5))
plt.hist(df["SNOW"], bins=80, edgecolor="black")
plt.title("Distribution of Daily Snowfall")
plt.xlabel("Snowfall (inches)")
plt.ylabel("Frequency")
plt.xlim(0, 30)
# plt.yscale('log')
plt.tight_layout()
plt.savefig("snowfall_histogram.png", dpi=300)
plt.show()

# ---------------------------------
# 12. Plots
# ---------------------------------
# First 300 test days
# Actual vs. predicted snowfall over time (time series plot)
plt.figure(figsize=(12, 6))
plt.plot(dates_test.values[:300], y_reg_test.values[:300], label="Actual")
plt.plot(dates_test.values[:300], y_pred_final[:300], label="Predicted")
plt.title("Snowfall Prediction (2-stage XGBoost model)")
plt.xlabel("Date")
plt.ylabel("Snowfall (inches)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("snowfall_prediction.png", dpi=300)
plt.show()

# Full time series
plt.figure(figsize=(14,6))
plt.plot(dates_test.values, y_reg_test.values, label="Actual")
plt.plot(dates_test.values, y_pred_final, label="Predicted")
plt.title("Full Snowfall Prediction (Test Set)")
plt.xlabel("Date")
plt.ylabel("Snowfall (inches)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("snowfall_prediction_full.png", dpi=300)
plt.show()

# Scatter plot (actual vs. predicted)
plt.figure(figsize=(7, 7))
plt.scatter(
    y_reg_test,
    y_pred_final,
    c=y_reg_test,   # color by actual snowfall
    cmap="coolwarm",
    alpha=0.5
)
plt.xlabel("Actual Snowfall (inches)")
plt.ylabel("Predicted Snowfall (inches)")
plt.title("Actual vs Predicted Snowfall (colored by magnitude)")
plt.savefig("actual_vs_predicted_scatter.png", dpi=300) # save scatter plot 
plt.tight_layout()
plt.show()

# Probability time series 
plt.figure(figsize=(12, 4))
plt.plot(dates_test.values[:300], y_clf_prob[:300])
plt.axhline(threshold, linestyle="--")
plt.title("Predicted Probability of Snow Tomorrow")
plt.xlabel("Date")
plt.ylabel("Probability")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_clf_test, y_clf_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["No Snow", "Snow"],
    yticklabels=["No Snow", "Snow"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Snow vs No Snow)")
plt.savefig("confusion_matrix.png", dpi=300) # save confusion matrix
plt.show()

print("Actual snow days in test set:", int(y_clf_test.sum()))
print("Predicted snow days in test set:", int(y_clf_pred.sum()))

# Threshold tuning  
for t in [0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78]:
    preds = (y_clf_prob > t).astype(int)
    precision = precision_score(y_clf_test, preds, zero_division=0)
    recall = recall_score(y_clf_test, preds, zero_division=0)
    f1 = f1_score(y_clf_test, preds, zero_division=0)
    print(f"Threshold {t}: snow days={preds.sum()}, precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")

# ---------------------------------
# Model Comparison
# ---------------------------------
print("\nMODEL COMPARISON\n")

# ---------------------------------
# Linear Regression
# ---------------------------------
lin_model = LinearRegression()
lin_model.fit(X_train, y_reg_train) # train 

y_pred_lin = lin_model.predict(X_test) # predict on test set 

# clip negatives
y_pred_lin[y_pred_lin < 0] = 0

rmse_lin = mean_squared_error(y_reg_test, y_pred_lin) ** 0.5
mae_lin = mean_absolute_error(y_reg_test, y_pred_lin)

print("Linear Regression:")
print("RMSE:", round(rmse_lin, 4))
print("MAE :", round(mae_lin, 4))

# ---------------------------------
# Random Forest
# ---------------------------------
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_reg_train)

y_pred_rf = rf_model.predict(X_test)

# clip small noise
y_pred_rf[y_pred_rf < 0.1] = 0

rmse_rf = mean_squared_error(y_reg_test, y_pred_rf) ** 0.5
mae_rf = mean_absolute_error(y_reg_test, y_pred_rf)

print("\nRandom Forest:")
print("RMSE:", round(rmse_rf, 4))
print("MAE :", round(mae_rf, 4))

# ---------------------------------
# Compare with XGBoost model
# ---------------------------------
print("\nXGBoost (2-stage model):")
print("RMSE:", round(rmse, 4))
print("MAE :", round(mae, 4))

# ---------------------------------
# Summary table
# ---------------------------------
print("\n========== FINAL COMPARISON ==========")
print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10}")
print(f"{'Linear Regression':<20} {rmse_lin:<10.2f} {mae_lin:<10.2f}")
print(f"{'Random Forest':<20} {rmse_rf:<10.2f} {mae_rf:<10.2f}")
print(f"{'XGBoost (2-stage)':<20} {rmse:<10.2f} {mae:<10.2f}")

# ---------------------------------
# Average yearly snowfall (inches)
# ---------------------------------

# Extract year
df["year"] = df["DATE"].dt.year

# Total snowfall per year
yearly_totals = df.groupby("year")["SNOW"].sum()

# Average across years
avg_yearly_snow = yearly_totals.mean()

print("\nAverage yearly snowfall (Grand Targhee SNOTEL estimated snowfall):")
print(f"{avg_yearly_snow:.2f} inches")