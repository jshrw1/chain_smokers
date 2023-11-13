#  1. Define problem: Binary classifier model that predicts smoker status. Smoker = 1.

#  2. Collect Data: Data collated by Kaggle as part of competition. No further data sourced.

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

#  Compute Additional Features
