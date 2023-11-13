#  1. Define problem: Binary classifier model that predicts smoker status. Smoker = 1.

#  2. Collect Data: Data collated by Kaggle as part of competition. No further data sourced.
import numpy as np
import pandas as pd

# Load training data into a pandas DataFrame
train_path = r'~/PycharmProjects/chain_smokers/data/train.csv'
test_path = r'~/PycharmProjects/chain_smokers/data/train.csv'
train = pd.read_csv(train_path, header=0)
test = pd.read_csv(test_path, header=0)

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

# BMI - combines height and weight. Obesity predictor
train['bmi'] = train['weight(kg)']/(train['height(cm)']/100*train['height(cm)']/100)

# Waist to height ratio. Additional predictor of obesity
train['wst_hgt'] = train['waist(cm)'] / train['height(cm)']

#  Blood pressure - relevant for systolic #0 - low #1 - normal #2pre-high #4-high
train['bld_pres'] = pd.cut(train['systolic'], bins=[0, 90, 120, 140, np.inf], labels=False, right=False)

# Diabetic - relevant to the fasting blood sugar
train['diabetic'] = pd.cut(train['fasting blood sugar'], bins=[0, 99, 125, np.inf], labels=False, right=False)

#  Cholesterol risk factors
train['tc'] = train['HDL'] + train['LDL'] + 0.2*train['triglyceride']
train['tc'] = pd.cut(train['tc'], bins=[0, 200, 239, np.inf], labels=False, right=False)
train['ldl'] = pd.cut(train['LDL'], bins=[0, 129, 159, np.inf], labels=False, right=False)
train['hdl'] = pd.cut(train['HDL'], bins=[0, 39, 49, np.inf], labels=False, right=False)
train['trig'] = pd.cut(train['triglyceride'], bins=[0, 200, 399, np.inf], labels=False, right=False)

# Risk factors related to hemoglobin
train['red_blood'] = pd.cut(train['hemoglobin'], bins=[0, 12, 17.5, np.inf], labels=False, right=False)

#  Kidney Risk factor related to urine protein and serum creatinine
# Liver risk factors AST, ALT, Gtp

