import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cs-training.csv', index_col=0)

from sklearn.model_selection import train_test_split

train_val_df, test_df = train_test_split(
    df, test_size=0.15, stratify=df['SeriousDlqin2yrs'], random_state=42
)

# Step 2: split remaining 85% into train (70% of total) and val (15% of total)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.1765, stratify=train_val_df['SeriousDlqin2yrs'], random_state=42
)

# ... imports, data loading, splitting (already done) ...

class CreditDataCleaner:
    def __init__(self):
        self.income_median_ = None
        self.dependents_median_ = None
        self.revolving_cap_ = None
        self.debtratio_cap_ = None

    def fit(self, df):
        self.income_median_ = df['MonthlyIncome'].median()
        self.dependents_median_ = df['NumberOfDependents'].median()
        self.revolving_cap_ = df['RevolvingUtilizationOfUnsecuredLines'].quantile(0.99)
        self.debtratio_cap_ = df['DebtRatio'].quantile(0.99)
        return self

    def transform(self, df):
        df = df.copy()
        df = df[df['age'] > 0].reset_index(drop=True)

        df['MonthlyIncome_missing'] = df['MonthlyIncome'].isnull().astype(int)
        df['NumberOfDependents_missing'] = df['NumberOfDependents'].isnull().astype(int)

        df['MonthlyIncome'] = df['MonthlyIncome'].fillna(self.income_median_)
        df['NumberOfDependents'] = df['NumberOfDependents'].fillna(self.dependents_median_)

        delinq_cols = ['NumberOfTime30-59DaysPastDueNotWorse',
                       'NumberOfTimes90DaysLate',
                       'NumberOfTime60-89DaysPastDueNotWorse']

        sentinel_mask = pd.Series(False, index=df.index)
        for col in delinq_cols:
            sentinel_mask = sentinel_mask | df[col].isin([96, 98])
        df['HadExtremeDelinquencyCode'] = sentinel_mask.astype(int)

        for col in delinq_cols:
            df[col] = df[col].clip(upper=20)

        df['TotalTimesPastDue'] = df[delinq_cols].sum(axis=1)

        df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=self.revolving_cap_)
        df['DebtRatio'] = df['DebtRatio'].clip(upper=self.debtratio_cap_)

        df['CreditLinesPerAge'] = df['NumberOfOpenCreditLinesAndLoans'] / df['age']
        df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
        df['HasRealEstateLoan'] = (df['NumberRealEstateLoansOrLines'] > 0).astype(int)

        return df

from sklearn.preprocessing import StandardScaler

class FeatureScaler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = None

    def fit(self, df, target_col='SeriousDlqin2yrs'):
        self.feature_cols = df.drop(columns=[target_col]).columns
        self.scaler.fit(df[self.feature_cols])
        return self

    def transform(self, df, target_col='SeriousDlqin2yrs'):
        result = df.copy()
        result[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return result

cleaner = CreditDataCleaner()
cleaner.fit(train_df)
train_clean = cleaner.transform(train_df)
val_clean = cleaner.transform(val_df)
test_clean = cleaner.transform(test_df)


scaler = FeatureScaler()
scaler.fit(train_clean)

train_scaled = scaler.transform(train_clean)
val_scaled = scaler.transform(val_clean)
test_scaled = scaler.transform(test_clean)

train_scaled.to_csv('data_train_scaled.csv', index=False)
val_scaled.to_csv('data_val_scaled.csv', index=False)
test_scaled.to_csv('data_test_scaled.csv', index=False)