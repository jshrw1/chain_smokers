#  1. Define problem: Binary classifier model that predicts smoker status. Smoker = 1.

#  2. Collect Data: Data collated by Kaggle as part of competition. No further data sourced.

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load training data into a pandas DataFrame
train_path = r'~/PycharmProjects/chain_smokers/data/train.csv'
test_path = r'~/PycharmProjects/chain_smokers/data/test.csv'
_train = pd.read_csv(train_path, header=0)


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
a = summary(_train)

#  4. EdA and Feature Engineering

#  Variables are relatively self-explanatory in how they might affect smoking. i.e. higher risk factors, higher blood pressure/ higher chloestrol etc.
#  So will focus on creating additional features based on data we have.

train = pd.DataFrame()

#  Age
train['age'] = _train['age']
#  Resting heartrate
train['rh'] = _train['relaxation']
# Tooth Decay
train['tooth'] = _train['dental caries']

#  Eyesight & Hearing
train['eyes'] = (_train['eyesight(left)'] + _train['eyesight(right)']) / 2
train['hearing'] = (_train['hearing(left)'] + _train['hearing(right)']) / 2

# BMI - combines height and weight. Obesity predictor
train['bmi'] = _train['weight(kg)'] / (_train['height(cm)'] / 100 * _train['height(cm)'] / 100)
train['bmi_bin'] = pd.cut(train['bmi'], bins=[0, 18, 25, 30, np.inf], labels=False, right=False)

# Waist to height ratio. Additional predictor of obesity
train['wst_hgt'] = _train['waist(cm)'] / _train['height(cm)']

#  Blood pressure - relevant for systolic #0 - low #1 - normal #2pre-high #4-high
train['bld_pres_bin'] = pd.cut(_train['systolic'], bins=[0, 90, 120, 140, np.inf], labels=False, right=False)

# Diabetic - relevant to the fasting blood sugar
train['diabetic_bin'] = pd.cut(_train['fasting blood sugar'], bins=[0, 99, 125, np.inf], labels=False, right=False)

#  Cholesterol risk factors
train['tc'] = _train['HDL'] + _train['LDL'] + 0.2 * _train['triglyceride']
train['tc_bin'] = pd.cut(train['tc'], bins=[0, 200, 239, np.inf], labels=False, right=False)
train['ldl_bin'] = pd.cut(_train['LDL'], bins=[0, 129, 159, np.inf], labels=False, right=False)
train['hdl_bin'] = pd.cut(_train['HDL'], bins=[0, 39, 49, np.inf], labels=False, right=False)
train['trig_bin'] = pd.cut(_train['triglyceride'], bins=[0, 200, 399, np.inf], labels=False, right=False)

# Risk factors related to hemoglobin
train['red_blood_bin'] = pd.cut(_train['hemoglobin'], bins=[0, 12, 17.5, np.inf], labels=False, right=False)

#  Correlation matrix
f, ax = plt.subplots(figsize=(10, 8))
corr = train.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

#  5. Estimate baseline model
#  Varius models to select from: Logit, Decision Tree, Random Forest, SVM, Naive Bays, KNN, Neural Networks, Gradient Boosting (XGBoost, Adaboost

features = list(train.columns)
X = train[features]
y = _train['smoking']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

target_names = ['no smoker', 'smoker']
print(classification_report(y_test, y_pred, target_names=target_names))

y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()