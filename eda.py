#  1. Define problem: Binary classifier model that predicts smoker status. Smoker = 1.
#  2. Collect Data: Data collated by Kaggle as part of competition. No further data sourced.
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Load training data into a pandas DataFrame
train_path = r'~/PycharmProjects/chain_smokers/data/train.csv'
test_path = r'~/PycharmProjects/chain_smokers/data/test.csv'
train = pd.read_csv(train_path, header=0)


#  3. Data processing: Missing, Encoding, Imputation, Standardisation and Normalisation.
#  No missing data, No variables to be encoded or Imputed.
#  Logistic regression/ Binary decision trees models do not require standardisation but may require normalisation
#  (Assumption on data having a gaussian distribution) Will leave the data alone apart from some feature engineering.
def clean_col_names(df):
    result_list = []
    seen_items = {}

    for i, item in enumerate(df.columns):
        match = re.search(r".*(?=[(\[])", item)
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

#  4. EdA and Feature Engineering
#  Variables are relatively self-explanatory in how they might affect smoking. i.e. higher risk factors,
#  higher blood pressure/ higher cholesterol etc. So will focus on creating additional features based on data we have.

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

a = summary(train)

#  5. Feature Selection
#  Small dataset-only 22 dependant variables-unlucky to need to do any unsupervised techniques i.e. PCA, ICA
#  to reduce dimensionality

#  Information Gain
# Set features for X and y
features = train.columns.tolist()
features.remove('smoking')

# Split the dataset into features and target
X = train[features]
y = train['smoking']

# Apply Information Gain
ig = mutual_info_regression(X, y)

# Create a dictionary of feature importance scores
feature_scores = {}
for i in range(len(features)):
    feature_scores[features[i]] = ig[i]
# Sort the features by importance score in descending order
sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

# Print the feature importance scores and the sorted features
for feature, score in sorted_features:
    print("Feature:", feature, "Score:", score)

#  Backward Selection

# Split the dataset into features and target
X = train[features]
y = train['smoking']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define the logistic regression model
model = LogisticRegression(max_iter=500)

# Define the forward selection object
sfs = SFS(model, k_features=10, forward=True, floating=False, scoring="accuracy", cv=5)

# Perform forward selection on the training set
sfs.fit(X_train, y_train)

# Print the selected features
print("Selected Features:", sfs.k_feature_names_)

# Evaluate the performance of the selected features on the testing set
accuracy = sfs.k_score_
print("Accuracy:", accuracy)

# Plot the performance of the model with different feature subsets
sfs_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
sfs_df["avg_score"] = sfs_df["avg_score"].astype(float)
fig, ax = plt.subplots()
sfs_df.plot(kind="line", y="avg_score", ax=ax)
ax.set_xlabel("Number of Features")
ax.set_ylabel("Accuracy")
ax.set_title("Forward Selection Performance")
plt.show()

features = list(sfs.k_feature_names_)

X = train[features]
y = train['smoking']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

# Model Validation and scoring
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

test_path = r'~/PycharmProjects/chain_smokers/data/test.csv'
test = pd.read_csv(test_path, header=0)
test = clean_col_names(test)

#  4. EdA and Feature Engineering
#  Variables are relatively self-explanatory in how they might affect smoking. i.e. higher risk factors,
#  higher blood pressure/ higher cholesterol etc. So will focus on creating additional features based on data we have.

#  Eyesight & Hearing
test['eyes_'] = (test['eyesight'] + test['eyesight_1']) / 2
test['hearing_'] = (test['hearing'] + test['hearing_1']) / 2
test = test.drop(['eyesight', 'eyesight_1', 'hearing_1', 'hearing'], axis=1)

# BMI - combines height and weight. Obesity predictor
test['bmi'] = test['weight'] / (test['height'] / 100 * test['height'] / 100)

# Waist to height ratio. Additional predictor of obesity
test['wst_hgt'] = test['waist'] / test['height']

#  Cholesterol risk factors
test['tc'] = test['hdl'] + test['ldl'] + 0.2 * test['triglyceride']
test = test.drop(['cholesterol'], axis=1)

X_test = test[features]
test['smoking'] = logreg.predict_proba(X_test)[::, 1]

submit = test[['id', 'smoking']]
submit.to_csv('submission.csv', index=False)
