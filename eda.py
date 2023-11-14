#  1. Define problem: Binary classifier model that predicts smoker status. Smoker = 1.

#  2. Collect Data: Data collated by Kaggle as part of competition. No further data sourced.

import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load training data into a pandas DataFrame
train_path = r'~/PycharmProjects/chain_smokers/data/train.csv'
test_path = r'~/PycharmProjects/chain_smokers/data/test.csv'
train = pd.read_csv(train_path, header=0)


#  3. Data processing: Missing, Encoding, Imputation, Standardisation and Normalisation.
#  No missing data, No variables to be encoded or Imputed.
#  Logistic regression/ Binary decision trees models do not require standardisation but may require normalisation (Assumption on data having a gaussian distribution)
#  Will leave the data alone apart from some feature engineering.


def summary(df):
    print(f'data shape: {df.shape}')
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values
    summ['%missing'] = df.isnull().sum().values / len(df) * 100
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['mean'] = desc['mean'].values
    summ['median'] = desc['50%'].values
    summ['standard_deviation'] = desc['std'].values
    summ['first value'] = df.loc[0].values
    summ['second value'] = df.loc[1].values
    summ['third value'] = df.loc[2].values

    return summ


#  Summary table of variables
a = summary(train)

#  4. EdA and Feature Engineering

#  Variables are relatively self-explanatory in how they might affect smoking. i.e. higher risk factors, higher blood pressure/ higher chloestrol etc.
#  So will focus on creating additional features based on data we have.

#  Eyesight & Hearing
train['eyes'] = (train['eyesight(left)'] + train['eyesight(right)']) / 2
train['hearing'] = (train['hearing(left)'] + train['hearing(right)']) / 2

# BMI - combines height and weight. Obesity predictor
train['bmi'] = train['weight(kg)'] / (train['height(cm)'] / 100 * train['height(cm)'] / 100)
train['bmi_bin'] = pd.cut(train['bmi'], bins=[0, 18, 25, 30, np.inf], labels=False, right=False)

# Waist to height ratio. Additional predictor of obesity
train['wst_hgt'] = train['waist(cm)'] / train['height(cm)']

#  Blood pressure - relevant for systolic #0 - low #1 - normal #2pre-high #4-high
train['bld_pres_bin'] = pd.cut(train['systolic'], bins=[0, 90, 120, 140, np.inf], labels=False, right=False)

# Diabetic - relevant to the fasting blood sugar
train['diabetic_bin'] = pd.cut(train['fasting blood sugar'], bins=[0, 99, 125, np.inf], labels=False, right=False)

#  Cholesterol risk factors
train['tc'] = train['HDL'] + train['LDL'] + 0.2 * train['triglyceride']
train['tc_bin'] = pd.cut(train['tc'], bins=[0, 200, 239, np.inf], labels=False, right=False)
train['ldl_bin'] = pd.cut(train['LDL'], bins=[0, 129, 159, np.inf], labels=False, right=False)
train['hdl_bin'] = pd.cut(train['HDL'], bins=[0, 39, 49, np.inf], labels=False, right=False)
train['trig_bin'] = pd.cut(train['triglyceride'], bins=[0, 200, 399, np.inf], labels=False, right=False)

# Risk factors related to hemoglobin
train['red_blood_bin'] = pd.cut(train['hemoglobin'], bins=[0, 12, 17.5, np.inf], labels=False, right=False)

#  Correlation matrix
# f, ax = plt.subplots(figsize=(10, 8))
# corr = train.corr()
# sns.heatmap(corr,
#            xticklabels=corr.columns.values,
#            yticklabels=corr.columns.values)

#  5. Estimate baseline model
#  Varius models to select from: Logit, Decision Tree, Random Forest, SVM, Naive Bays, KNN, Neural Networks, Gradient Boosting (XGBoost, Adaboost

# Remover insignificant VAR and collinear vars
features = list(train.columns)
features.remove('smoking')
features.remove('id')
features.remove('eyesight(left)')
features.remove('eyesight(right)')
features.remove('hearing(left)')
features.remove('hearing(right)')
features.remove('serum creatinine')
features.remove('triglyceride')
features.remove('HDL')
features.remove('LDL')
features.remove('tc_bin')
features.remove('age')
features.remove('Cholesterol')
features.remove('hearing')
features.remove('bld_pres_bin')
features.remove('wst_hgt')
features.remove('eyes')

X = train[features]
y = train['smoking']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

# Model Validaiton and scoring
logreg = LogisticRegression(max_iter=1000)

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
target_names = ['no smoker', 'smoker']
print(classification_report(y_test, y_pred, target_names=target_names))
y_pred_proba = logreg.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()

# Submission:
test_path = r'~/PycharmProjects/chain_smokers/data/test.csv'
test = pd.read_csv(test_path, header=0)

#  Eyesight & Hearing
test['eyes'] = (test['eyesight(left)'] + test['eyesight(right)']) / 2
test['hearing'] = (test['hearing(left)'] + test['hearing(right)']) / 2

# BMI - combines height and weight. Obesity predictor
test['bmi'] = test['weight(kg)'] / (test['height(cm)'] / 100 * test['height(cm)'] / 100)
test['bmi_bin'] = pd.cut(test['bmi'], bins=[0, 18, 25, 30, np.inf], labels=False, right=False)

# Waist to height ratio. Additional predictor of obesity
test['wst_hgt'] = test['waist(cm)'] / test['height(cm)']

#  Blood pressure - relevant for systolic #0 - low #1 - normal #2pre-high #4-high
test['bld_pres_bin'] = pd.cut(test['systolic'], bins=[0, 90, 120, 140, np.inf], labels=False, right=False)

# Diabetic - relevant to the fasting blood sugar
test['diabetic_bin'] = pd.cut(test['fasting blood sugar'], bins=[0, 99, 125, np.inf], labels=False, right=False)

#  Cholesterol risk factors
test['tc'] = test['HDL'] + test['LDL'] + 0.2 * test['triglyceride']
test['tc_bin'] = pd.cut(test['tc'], bins=[0, 200, 239, np.inf], labels=False, right=False)
test['ldl_bin'] = pd.cut(test['LDL'], bins=[0, 129, 159, np.inf], labels=False, right=False)
test['hdl_bin'] = pd.cut(test['HDL'], bins=[0, 39, 49, np.inf], labels=False, right=False)
test['trig_bin'] = pd.cut(test['triglyceride'], bins=[0, 200, 399, np.inf], labels=False, right=False)

# Risk factors related to hemoglobin
test['red_blood_bin'] = pd.cut(test['hemoglobin'], bins=[0, 12, 17.5, np.inf], labels=False, right=False)

features = list(test.columns)
features.remove('id')
features.remove('eyesight(left)')
features.remove('eyesight(right)')
features.remove('hearing(left)')
features.remove('hearing(right)')
features.remove('serum creatinine')
features.remove('triglyceride')
features.remove('HDL')
features.remove('LDL')
features.remove('tc_bin')
features.remove('age')
features.remove('Cholesterol')
features.remove('hearing')
features.remove('bld_pres_bin')
features.remove('wst_hgt')
features.remove('eyes')
X_test = test[features]
test['smoking'] = logreg.predict_proba(X_test)[::, 1]

submit = test[['id', 'smoking']]
submit.to_csv('submission.csv', index=False)