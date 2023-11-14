#  1. Define problem: Binary classifier model that predicts smoker status. Smoker = 1.
#  2. Collect Data: Data collated by Kaggle as part of competition. No further data sourced.
import re
import pandas as pd
import seaborn as sns

# Load training data into a pandas DataFrame
train_path = r'~/PycharmProjects/chain_smokers/data/train.csv'
test_path = r'~/PycharmProjects/chain_smokers/data/test.csv'
train = pd.read_csv(train_path, header=0)


#  3. Data processing: Missing, Encoding, Imputation, Standardisation and Normalisation.
#  No missing data, No variables to be encoded or Imputed.
#  Logistic regression/ Binary decision trees models do not require standardisation but may require normalisation (Assumption on data having a gaussian distribution)
#  Will leave the data alone apart from some feature engineering.
def clean_col_names(df):
    result_list = []
    seen_items = {}

    for i, item in enumerate(df.columns):
        match = re.search(r".*(?=[\(\[])", item)
        if match:
            text_before_bracket = match.group(0)
        else:
            text_before_bracket = item

        # Replace spaces with underscores and convert to lowercase
        text_before_bracket = text_before_bracket.replace(' ', '_').lower()

        # Check for duplicates and append "_1", "_2", etc.
        if text_before_bracket in seen_items:
            seen_items[text_before_bracket] += 1
            text_before_bracket += f"_{seen_items[text_before_bracket]}"
        else:
            seen_items[text_before_bracket] = 0

        result_list.append(text_before_bracket)

    df.columns = result_list
    return df


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


train = clean_col_names(train)
a = summary(train)

#  4. EdA and Feature Engineering
#  Variables are relatively self-explanatory in how they might affect smoking. i.e. higher risk factors, higher blood pressure/ higher chloestrol etc.
#  So will focus on creating additional features based on data we have.

#  Eyesight & Hearing
train['eyes_'] = (train['eyesight'] + train['eyesight_1']) / 2
train['hearing_'] = (train['hearing'] + train['hearing_1']) / 2
train = train.drop(['eyesight', 'eyesight_1', 'hearing_1', 'hearing'], axis=1)

# BMI - combines height and weight. Obesity predictor
train['bmi'] = train['weight'] / (train['height'] / 100 * train['height'] / 100)

# Waist to height ratio. Additional predictor of obesity
train['wst_hgt'] = train['waist'] / train['height']

#  Cholesterol risk factors
train['tc'] = train['hdl'] + train['ldl'] + 0.2 * train['triglyceride']
train = train.drop(['cholesterol'], axis=1)

#  Correlation matrix
train = train.drop(['id'], axis=1)
corr = train.corr()
sns.heatmap(corr, cmap='Spectral', vmin=-1, vmax=1, square=True)

#  5. Feature Selection
#  Small dataset-only 22 dependant variables-unlucky to need to do any unsupervised techniques i.e. PCA, ICA
#  to reduce dimensionality

#  Information Gain
#  Backward Selection
#  Regularization