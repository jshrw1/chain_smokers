# 5. Estimate baseline model
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