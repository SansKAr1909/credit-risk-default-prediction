import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

target_col = 'SeriousDlqin2yrs'

# ============================================================
# LOAD MODEL AND DATA
# ============================================================
model_xgb = joblib.load('final_xgb_model.pkl')
test_clean = pd.read_csv('data_test_clean.csv')
X_test = test_clean.drop(columns=[target_col])

# Recompute predicted probabilities
test_probs = model_xgb.predict_proba(X_test)[:, 1]

# ============================================================
# SHAP VALUES
# ============================================================
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test)

# ============================================================
# GLOBAL SUMMARY PLOT
# ============================================================
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()   # this actually opens the window

# ============================================================
# INDIVIDUAL EXPLANATION
# ============================================================
risky_idx = X_test[test_probs > 0.5].index[0]
row_position = X_test.index.get_loc(risky_idx)

waterfall_exp = shap.Explanation(
    values=shap_values[row_position],
    base_values=explainer.expected_value,
    data=X_test.loc[risky_idx],
    feature_names=X_test.columns.tolist()
)

shap.plots.waterfall(waterfall_exp, show=False)
plt.tight_layout()
plt.savefig('shap_waterfall_example.png', dpi=150, bbox_inches='tight')
plt.show()