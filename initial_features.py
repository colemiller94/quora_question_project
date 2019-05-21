import time
from fuzzywuzzy import fuzz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm

#basic features
df_200K['len_q1'] = df_200K['question1'].apply(lambda x: len(str(x)))
df_200K['len_q2'] = df_200K['question2'].apply(lambda x: len(str(x)))
df_200K['diff_len'] = abs(df_200K['len_q1'] - df_200K['len_q2'])
df_200K['len_char_q1'] = df_200K['question1'].apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
df_200K['len_char_q2'] = df_200K['question2'].apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
df_200K['len_word_q1'] = df_200K['question1'].apply(lambda x: len(str(x).split()))
df_200K['len_word_q2'] = df_200K['question2'].apply(lambda x: len(str(x).split()))
df_200K['common_words'] = df_200K.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)

#fuzzywuzzy ratio features
df_200K['fuzz_simple_ratio'] = df_200K.apply(lambda x: fuzz.ratio(str(x['question1']), str(x['question2'])), axis=1)
df_200K['fuzz_partial_ratio'] = df_200K.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
df_200K['fuzz_token_set_ratio'] = df_200K.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
df_200K['fuzz_partial_token_set_ratio'] = df_200K.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
df_200K['fuzz_token_sort_ratio'] = df_200K.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
df_200K['fuzz_partial_token_sort_ratio'] = df_200K.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

#sim_by_verb_column
sim_by_verb_list = []
for i, j in zip(smart_df['q1'], smart_df['q2']):
    sim_by_verb_list.append(sim_by_pos(i,j,'verb'))
sim_by_verb_series = pd.Series(sim_by_verb_list)
sim_by_verb_series.rename("sim_by_verb", inplace=True)
smart_df = pd.concat([smart_df, sim_by_verb_series], axis=1)

#sim_of_diffs_by_verb_column
sim_of_diffs_by_verb_list = []
for i, j in zip(smart_df['q1'], smart_df['q2']):
    sim_of_diffs_by_verb_list.append(sim_of_diffs_by_pos(i,j,'verb'))
sim_of_diffs_by_verb_series = pd.Series(sim_of_diffs_by_verb_list)
sim_of_diffs_by_verb_series.rename("sim_of_diffs_by_verb", inplace=True)
smart_df = pd.concat([smart_df, sim_of_diffs_by_verb_series], axis=1)

#sim_by_noun_column
sim_by_noun_list = []
for i, j in zip(smart_df['q1'], smart_df['q2']):
    sim_by_noun_list.append(sim_by_pos(i,j,'noun'))
sim_by_noun_series = pd.Series(sim_by_noun_list)
sim_by_noun_series.rename("sim_by_noun", inplace=True)
smart_df = pd.concat([smart_df, sim_by_noun_series], axis=1)

#sim_of_diffs_by_noun_column
sim_of_diffs_by_noun_list = []
for i, j in zip(smart_df['q1'], smart_df['q2']):
    sim_of_diffs_by_noun_list.append(sim_of_diffs_by_pos(i,j,'noun'))
sim_of_diffs_by_noun_series = pd.Series(sim_of_diffs_by_noun_list)
sim_of_diffs_by_noun_series.rename("sim_of_diffs_by_noun", inplace=True)
smart_df = pd.concat([smart_df, sim_of_diffs_by_noun_series], axis=1)

#distplots
sns.distplot(smart_df['sim_by_verb'])
sns.distplot(smart_df['sim_of_diffs_by_verb'])
sns.distplot(smart_df['sim_by_noun'])
sns.distplot(smart_df['sim_of_diffs_by_noun'])

#define X and y
X = df_200K.iloc[:,6:]
y = df_200K.iloc[:,5:6]

#transform
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

#logistic regression
logreg = LogisticRegression(fit_intercept = False, C = 1e12)
logreg_model = logreg.fit(X_train, y_train)
logreg_model
y_hat_train = logreg_model.predict(X_train)
y_hat_test = logreg_model.predict(X_test)
residuals = y_train_array - y_hat_train
pd.Series(residuals).value_counts()
pd.Series(residuals).value_counts(normalize=True)

y_test_array = np.array(y_test['is_duplicate'])
residuals = y_test_array - y_hat_test
pd.Series(residuals).value_counts()
pd.Series(residuals).value_counts(normalize=True)

print(confusion_matrix(y_test_array, y_hat_test))

#Random Forest
rf_clf = RandomForestClassifier()
mean_rf_cv_score = np.mean(cross_val_score(rf_clf, X_scaled, y, cv=3))
print("Mean Cross Validation Score for Random Forest Classifier: {:.4}%".format(mean_rf_cv_score * 100))

rf_param_grid = {
    'n_estimators': [10, 30, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 6, 10],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [1, 2, 5]
}

start = time.time()
rf_grid_search = GridSearchCV(rf_clf, rf_param_grid, cv=3)
rf_grid_search.fit(X_scaled, y)

print("Testing Accuracy: {:.4}%".format(rf_grid_search.best_score_ * 100))
print("Total Runtime for Grid Search on Random Forest Classifier: {:.4} seconds".format(time.time() - start))
print("")
print("Optimal Parameters: {}".format(rf_grid_search.best_params_))

#SVM

import time
start_time = time.time()
clf = svm.SVC(probability=True)
clf.fit(X_train, y_train)
total = time.time() - start_time
clf.predict_proba(X_test)
clf.score(X_test, y_test)

