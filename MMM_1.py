import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

def mmm_data_quality_check(df, revenue_col, media_cols, control_candidates):

    #Run this before fitting any MMM model.
    print("MMM DATA QUALITY CHECKLIST")

    #1: Missing values 
    print("1. MISSING VALUES")
    missing = df[media_cols + control_candidates + [revenue_col]].isnull().sum()
    if missing.sum() == 0:
        print("PASS- No missing values found")
    else:
        print("WARNING- Missing values found:")
        print(missing[missing > 0])

    #2: Zero spend weeks per channel
    print("2. ZERO SPEND WEEKS (channels that are often off)")
    for col in media_cols:
        zero_pct = (df[col] == 0).mean() * 100
        status = "WARNING" if zero_pct > 80 else "OK"
        print(f" {col:<25} {zero_pct:.1f}% zero weeks  [{status}]")
    print("High zero % = sparse spend = harder to identify effect")

    # 3: Correlation of controls with revenue
    print("3. CONTROL VARIABLE CORRELATION WITH REVENUE")
    print("Above 0.95 = danger, it may dominate the model")
    for col in control_candidates:
        if col in df.columns:
            corr = df[col].corr(df[revenue_col])
            status = ("REMOVE" if abs(corr) > 0.98 else
                      "CAUTION" if abs(corr) > 0.90 else
                      "OK")
            print(f"{col:<25} corr={corr:.4f}  [{status}]")

    #4: Correlation between media channels
    print("4. INTER-CHANNEL CORRELATION (multicollinearity risk)")
    print("Above 0.7 = high multicollinearity risk")
    corr_matrix = df[media_cols].corr()
    for i in range(len(media_cols)):
        for j in range(i+1, len(media_cols)):
            corr = corr_matrix.iloc[i, j]
            status = ("HIGH RISK" if abs(corr) > 0.7 else
                      "MODERATE" if abs(corr) > 0.4 else
                      "LOW")
            print(f"{media_cols[i]} vs {media_cols[j]:<20} "
                  f"corr={corr:.3f}  [{status}]")

    #5: Dataset length
    print("5. DATASET LENGTH")
    n_weeks = len(df)
    n_params = len(media_cols) + len(control_candidates) + 5 + 1

    ratio = n_weeks / n_params
    status = ("GOOD" if ratio >= 10 else
              "ACCEPTABLE" if ratio >= 5 else
              "WARNING — too few observations per parameter")
    print(f"Weeks: {n_weeks}")
    print(f"Estimated parameters: {n_params}")
    print(f"Ratio: {ratio:.1f} observations per parameter  [{status}]")
    print("Recommended minimum: 10 observations per parameter")

    print("CHECKLIST COMPLETE")

# Run on the dataset 
df = pd.read_csv("C:/Users/kaush/simulated_marketing_data.csv")
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)
print(f"Dataset: {df.shape[0]} weeks|"
      f"{df['DATE'].min().date()} to {df['DATE'].max().date()}")
print()

mmm_data_quality_check(
    df            = df,
    revenue_col   = 'revenue',
    media_cols    = ['tv_S', 'radio_S', 'paid_search_S'],
    control_candidates = ['competitor_sales']
)

# ADSTOCK TRANSFORMATION
def adstock(series, decay):
    result = np.zeros(len(series))
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = series[i] + decay * result[i-1]
    return result

# Apply Adstock
df['tv_adstock']     = adstock(df['tv_S'].values, 0.6)
df['radio_adstock']  = adstock(df['radio_S'].values, 0.3)
df['search_adstock'] = adstock(df['paid_search_S'].values, 0.1)

#SATURATION (Michaelis-Menten)
# Models diminishing returns: S(x) = x / (x + K)
def saturation_mm(series):
    K = float(np.median(series[series > 0])) if (series > 0).any() else 1.0
    return series / (series + K) # K = median of non-zero adstock values per channel
 
df['tv_sat']     = saturation_mm(df['tv_adstock'])
df['radio_sat']  = saturation_mm(df['radio_adstock'])
df['search_sat'] = saturation_mm(df['search_adstock'])

# SEASONALITY AND TREND
df['month'] = df['DATE'].dt.month
month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True).astype(int)
df = pd.concat([df, month_dummies], axis=1)
month_cols = [c for c in df.columns if c.startswith('month_')]
df['trend'] = np.arange(len(df))
 
MEDIA    = ['tv_sat', 'radio_sat', 'search_sat']
feature_cols = MEDIA + ['trend'] + month_cols
y        = df['revenue']

# MULTICOLLINEARITY CHECK
media_cols = ['tv_sat', 'radio_sat', 'search_sat']

print("MULTICOLLINEARITY CHECK (VIF)")
print("VIF < 4 = fine | 4-10 = moderate | > 10 = serious")

vif = pd.DataFrame({
    'Variable': media_cols,
    'VIF': [variance_inflation_factor(
                df[media_cols].values, i)
            for i in range(len(media_cols))]
})
print(vif.to_string(index=False))
print()
print("Channel correlations:")
print(df[media_cols].corr().round(3))
print()

# FIT OLS MODEL
feature_cols = MEDIA + ['trend'] + month_cols

X = sm.add_constant(df[feature_cols])
y = df['revenue']
results = sm.OLS(y, X).fit()

print("OLS MODEL RESULTS")
print(f"R-squared: {results.rsquared:.4f}")
print(f"Adj R-squared: {results.rsquared_adj:.4f}")
print()

print("Channel Coefficients:")
for col in ['tv_sat', 'radio_sat', 'search_sat']:
    coef = results.pvalues[col]
    pval = results.pvalues[col]
    sig  = ("***" if pval < 0.001 else
            "**"  if pval < 0.01  else
            "*"   if pval < 0.05  else "n.s.")
    print(f"  {col:<25} coef={coef:>12.2f}   p={pval:.4f}  {sig}")

print()
print("*** = very confident this channel drives revenue")
print("n.s.= cannot confidently say this channel helps")

#Ridge
split       = int(len(df) * 0.8)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols].values)
ridge    = RidgeCV(alphas=[0.01,0.1,1,10,50,100,500], cv=5)
ridge.fit(X_scaled, y)
ridge_coefs = ridge.coef_ / scaler.scale_

y_pred_ridge = ridge.predict(X_scaled)
ridge_r2     = r2_score(y, y_pred_ridge)

X_tr = scaler.fit_transform(df[feature_cols].iloc[:split].values)
X_te = scaler.transform(df[feature_cols].iloc[split:].values)
ridge_tr = RidgeCV(alphas=[0.01,0.1,1,10,50,100,500], cv=5)
ridge_tr.fit(X_tr, y.iloc[:split])
ridge_pred = ridge_tr.predict(X_te)
ridge_mape = np.mean(np.abs(
    (y.iloc[split:].values - ridge_pred) / y.iloc[split:].values)) * 100

print(f"Ridge R-squared: {ridge_r2:.4f}")
print(f"Ridge OOS MAPE:  {ridge_mape:.2f}%")
print(f"Best alpha:      {ridge.alpha_}")

# CHANNEL CONTRIBUTION DECOMPOSITION
print()
print("CHANNEL CONTRIBUTION DECOMPOSITION")

contributions = {}
for col in media_cols:
    raw = (results.params[col] * df[col]).sum()
    contributions[col] = max(0, raw)

total_media   = sum(contributions.values())
total_revenue = y.sum()
baseline      = max(0, total_revenue - total_media)
grand_total   = baseline + total_media

for col, val in contributions.items():
    pct = val / grand_total * 100
    label = col.replace('_adstock', '').upper()
    print(f"  {label:<20} {pct:>6.1f}% of total revenue")

print(f"{'BASELINE':<20}"
      f"{baseline/grand_total*100:>6.1f}% of total revenue")
print()
print("Baseline = revenue that happens with zero marketing spend")
print("Media    = revenue directly caused by advertising")

# OUT-OF-SAMPLE VALIDATION
print()
print("OUT-OF-SAMPLE VALIDATION (80/20 split)")

split       = int(len(df) * 0.8)
X_train     = sm.add_constant(df[feature_cols].iloc[:split])
X_test      = sm.add_constant(df[feature_cols].iloc[split:])
y_train     = y.iloc[:split]
y_test      = y.iloc[split:]
m_train     = sm.OLS(y_train, X_train).fit()
y_pred      = m_train.predict(X_test)
mape        = np.mean(
    np.abs((y_test.values - y_pred.values) / y_test.values)
) * 100

print(f"Training: weeks 1 to {split}")
print(f"Testing:  weeks {split+1} to {len(df)}")
print(f"MAPE:     {mape:.2f}%")
print()
if   mape < 5:  print("EXCELLENT — within 5% accuracy on unseen data")
elif mape < 10: print("GOOD — within 10% accuracy on unseen data")
elif mape < 20: print("MODERATE — acceptable for first model")
else:           print("WEAK — model needs improvement")

# Ridge out-of-sample
X_train_r = scaler.fit_transform(df[feature_cols].iloc[:split].values)
X_test_r  = scaler.transform(df[feature_cols].iloc[split:].values)
ridge_oos = RidgeCV(alphas=[0.01,0.1,1,10,50,100,500], cv=5)
ridge_oos.fit(X_train_r, y_train)
y_pred_ridge_oos = ridge_oos.predict(X_test_r)
mape_ridge = np.mean(np.abs(
    (y_test.values - y_pred_ridge_oos) / y_test.values)) * 100

print(f"Ridge MAPE: {mape_ridge:.2f}%")
if   mape_ridge < 5:  print("EXCELLENT — within 5% accuracy on unseen data")
elif mape_ridge < 10: print("GOOD — within 10% accuracy on unseen data")
elif mape_ridge < 20: print("MODERATE — acceptable for first model")
else:                 print("WEAK — model needs improvement")

y_total = float(y.sum())

ols_c = {}
for col in media_cols:
    raw = (results.params[col] * df[col]).sum()
    ols_c[col] = max(0, raw)

ridge_c = {}
for i, col in enumerate(media_cols):
    raw = (ridge_coefs[i] * df[col]).sum()
    ridge_c[col] = max(0, raw)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('MMM Baseline — OLS vs Ridge', fontsize=13, fontweight='bold')

axes[0].plot(df['DATE'], y, label='Actual', linewidth=1.2, color='#2C3E50')
axes[0].plot(df['DATE'], results.fittedvalues,
             label=f'OLS (R²={results.rsquared:.3f})',
             linewidth=1.2, color='#E74C3C', linestyle='--')
axes[0].set_title('Actual vs Fitted Revenue')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=30)

labels     = [c.replace('_sat','').upper() for c in MEDIA]
ols_pcts   = [ols_c[c]/y_total*100   for c in MEDIA]
ridge_pcts = [ridge_c[c]/y_total*100 for c in MEDIA]
x = np.arange(len(labels))
axes[1].bar(x-0.2, ols_pcts,   0.35, label='OLS',   color='#E74C3C', alpha=0.8)
axes[1].bar(x+0.2, ridge_pcts, 0.35, label='Ridge', color='#3498DB', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_title('Channel Contributions')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

colors_vif = ['#E74C3C' if v>10 else '#F39C12' if v>4 else '#27AE60'
               for v in vif['VIF']]
axes[2].bar(vif['Variable'], vif['VIF'], color=colors_vif, width=0.5)
axes[2].axhline(y=4,  color='#F39C12', linestyle='--', label='VIF=4')
axes[2].axhline(y=10, color='#E74C3C', linestyle='--', label='VIF=10')
axes[2].set_title('VIF — Multicollinearity')
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('C:/Users/kaush/baseline_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: C:/Users/kaush/baseline_results.png")

print("FIRST MMM MODEL COMPLETE")