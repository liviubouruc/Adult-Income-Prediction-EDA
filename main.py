# ********************************************
# Preprocessing
# ********************************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from typing import List

@dataclass
class ColumnDesc:
    dtype: np.dtype
    values: List[str]

column_description = {
    'age': ColumnDesc(dtype=np.uint, values=None),
    'workclass': ColumnDesc(dtype=None, values=['Private', 'Self-emp-not-inc', 'Self-emp-inc', 
                                             'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']),
    'fnlwgt': ColumnDesc(dtype=np.uint, values=None),
    'education': ColumnDesc(dtype=None, values=['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', '5th-6th',
                                              'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate',  'Preschool']),
    'education-num': ColumnDesc(dtype=np.uint, values=None),
    'marital-status': ColumnDesc(dtype=None, values=['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 
                                                     'Married-spouse-absent', 'Married-AF-spouse']),
    'occupation': ColumnDesc(dtype=None, values=['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                                                  'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
                                                   'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']),
    'relationship': ColumnDesc(dtype=None, values=['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']),
    'race': ColumnDesc(dtype=None, values=['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']),
    'sex': ColumnDesc(dtype=None, values=['Female', 'Male']),
    'capital-gain': ColumnDesc(dtype=np.int, values=None),
    'hours-per-week': ColumnDesc(dtype=np.int, values=None),
    'native-country': ColumnDesc(dtype=None, values=['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                                                      'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China',
                                                       'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica',
                                                       'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 
                                                       'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 
                                                       'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago',
                                                        'Peru', 'Hong', 'Holand-Netherlands'])
}


df = pd.read_csv(r"C:\Users\rfrancu\Projects\eda\data\train.csv", index_col=False)
# strip leading and trailing whitespaces from each field
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
display(df.head())
display(df.info())

# Check for null values
print('Missing values')
print(df.isnull().sum())
# At first sight seems that doesn't contain null values

# Check for inconsistency based on column description
# Besides the description provided by the authors, the dataset seem to have ? values, instead of None
# so we will count them as missing values
missing_values = {}
for col in column_description.keys():
    if column_description[col].values is not None:
        missing_values[col] = len(df[~df[col].isin(column_description[col].values)])
    else:
        missing_values[col] = len(df[df[col] == '?'])

print('Missing values')
for key, val in missing_values.items():
    print(key, val)

# because we cannot use the rows with missing values or inconsistent values we will remove them
clean_df = df.copy()
for col in column_description.keys():
    if column_description[col].values is not None:
       clean_df = clean_df[clean_df[col].isin(column_description[col].values)]
    else:
        clean_df = clean_df[~(clean_df[col] == '?')]

print(len(df), len(clean_df))
clean_df.to_csv('clean_test.csv', index=False)

workclass_map = {
    'State-gov': 'sl-gov',
    'Local-gov': 'sl-gov',
    'Self-emp-not-inc': 'self-employed',
    'Self-emp-inc': 'self-employed',
    'Self-emp-inc': 'self-employed',
    'Without-pay': 'unemployed',
    'Never-worked': 'unemployed',
    'Private': 'private',
    'Federal-gov': 'federal-gov'
}

df['workclass'] = df['workclass'].apply(lambda x: workclass_map[x])
df['workclass'].unique()

north_america = ["Canada", "Cuba", "Dominican-Republic", "El-Salvador", "Guatemala",
                   "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua",
                   "Outlying-US(Guam-USVI-etc)", "Puerto-Rico", "Trinadad&Tobago",
                   "United-States"]

south_america = ["Columbia", "Ecuador", "Peru"]

asia = ["Cambodia", "China", "Hong", "India", "Iran", "Japan", "Laos",
        "Philippines", "Taiwan", "Thailand", "Vietnam"]

europe = ["England", "France", "Germany", "Greece", "Holand-Netherlands",
          "Hungary", "Ireland", "Italy", "Poland", "Portugal", "Scotland",
          "Yugoslavia"]

other = ['South']

native_country_map = {
    **{x: 'north_america' for x in north_america},
    **{x: 'south_america' for x in south_america},
    **{x: 'asia' for x in asia},
    **{x: 'europe' for x in europe},
    **{x: 'other' for x in other}
}

df['native-country'] = df['native-country'].apply(lambda x: native_country_map[x])
_, bins = np.histogram(df['age'], bins=10)
plt.hist(df['age'], bins=bins)
bins = [-1] + list(bins)
labels = list(range(len(bins) - 1))
df['age_cat'] = pd.cut(df['age'], bins=bins, labels=labels)

_, bins = np.histogram(df['capital-gain'], bins=3)
plt.hist(df['capital-gain'], bins=bins)
bins = [-1] + list(bins)
labels = list(range(len(bins) - 1))
df['capital-gain_cat'] = pd.cut(df['capital-gain'], bins=bins, labels=labels)

_, bins = np.histogram(df['capital-loss'], bins=3)
plt.hist(df['capital-loss'], bins=bins)
bins = [-1] + list(bins)
labels = list(range(len(bins) - 1))
df['capital-loss_cat'] = pd.cut(df['capital-loss'], bins=bins, labels=labels)

_, bins = np.histogram(df['hours-per-week'], bins=5)
plt.hist(df['hours-per-week'], bins=bins)
bins = [-1] + list(bins)
labels = list(range(len(bins) - 1))
df['hours-per-week_cat'] = pd.cut(df['hours-per-week'], bins=bins, labels=labels)

CAT_COLUMNS = ['workclass', 'education', 'marital-status', 'occupation', 
               'relationship', 'race', 'sex', 'native-country']
CAT_COLUMNS_alias = [x + '_cat' for x in CAT_COLUMNS]

le = LabelEncoder()
df[CAT_COLUMNS_alias] = df[CAT_COLUMNS].apply(le.fit_transform)

df['salary_label'] = df['salary'].apply(lambda x: 0 if x == '<=50K.' else 1)
df.to_csv('./data/processed_test.csv', index=False)

# ********************************************
# Logistic Regression - Francu Richard
# ********************************************

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
# from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import SelectFromModel
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import seaborn as sns
# from imblearn.over_sampling import SMOTE

# def evaluate(true, pred):
#     accuracy = accuracy_score(true, pred)
#     precision = precision_score(true, pred)
#     recall = recall_score(true, pred)
#     f1 = f1_score(true, pred)
#     print('Accuracy: %s' % accuracy)
#     print('Recall: %s' % recall)
#     print('Precision: %s' % precision)
#     print('F1: %s' % f1)
#     cm = confusion_matrix(true, pred)
#     sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'], fmt='5g')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.show()

# df = pd.read_csv(r'C:\Users\rfrancu\Projects\eda\data\processed_train.csv')
# df_test = pd.read_csv(r'C:\Users\rfrancu\Projects\eda\data\processed_test.csv')
# # Distribution of the target variable
# sns.countplot(x='salary_label', data=df)
# plt.show()

# # make a selection on features and compute the correlation matrix
# # V1
# FEATURE_COLUMN = ['age_cat', 'workclass_cat', 'education_cat', 'education-num', 'marital-status_cat', 'race_cat', 
#                   'sex_cat', 'hours-per-week_cat', 'capital-gain', 'capital-gain_cat', 'native-country_cat', 
#                   'relationship_cat', 'hours-per-week', 'occupation_cat']

# # V2, after feature selection based on model coeficients
# # FEATURE_COLUMN = ['capital-gain', 'marital-status_cat', 'education-num', 'age_cat',
# #                   'hours-per-week', 'sex_cat', 'race_cat', 'workclass_cat']

# TARGET_COLUMN = ['salary_label']

# df = df[FEATURE_COLUMN + TARGET_COLUMN]
# df_test = df_test[FEATURE_COLUMN + TARGET_COLUMN]
# corr_matrix = df.corr()
# fig, axs = plt.subplots(figsize=(10, 10))
# sns.heatmap(corr_matrix, annot=True, ax=axs)
# plt.show()

# X_train = df[FEATURE_COLUMN]
# y_train = df[TARGET_COLUMN]
# X_test = df_test[FEATURE_COLUMN]
# y_test = df_test[TARGET_COLUMN]

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # define model
# model = LogisticRegression(random_state=42, max_iter=1000)
# # fit it
# model.fit(X_train_scaled, y_train)
# y_train_pred = model.predict(X_train_scaled)
# # test
# y_test_pred = model.predict(X_test_scaled)
# # performance
# evaluate(y_train, y_train_pred)
# evaluate(y_test, y_test_pred)

# # Get feature importance based on model coeficients

# coefficients = model.coef_

# # Create a dataframe of the feature importance
# feature_importance = pd.DataFrame(coefficients[0], X_train.columns, columns=['importance'])
# feature_importance['importance'] = feature_importance['importance'].abs()
# feature_importance.sort_values(by='importance', ascending=False, inplace=True)

# # Print top 10 most important features
# print(feature_importance.head(12))

# # Apply SMOTE oversampling
# smote = SMOTE(random_state=42)
# X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)


# # Train a logistic regression model on the oversampled data
# model = LogisticRegression()
# model.fit(X_train_sm, y_train_sm)

# # Make predictions on the test data
# y_train_pred = model.predict(X_train_scaled)
# y_test_pred = model.predict(X_test_scaled)

# evaluate(y_train, y_train_pred)
# evaluate(y_test, y_test_pred)

# # Define the hyperparameters and their possible values
# parameters = {
#     'C':[0.01,0.1,1,10], 
#     'penalty':['l1','l2', 'elasticnet'], 
#     'solver': ['liblinear', 'lbfgs']
#     }

# # Define the logistic regression model
# log_reg = LogisticRegression(random_state=42, max_iter=1000)

# # Define the cross-validation method
# cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

# # Create a GridSearchCV object
# clf = GridSearchCV(log_reg, parameters, cv=cv, scoring='f1')

# # Fit the GridSearchCV object to the data
# clf.fit(X_train_sm, y_train_sm)

# # Print the best hyperparameters
# print("Best Hyperparameters : " + str(clf.best_params_))

# # Perform classification using best found parameters
# config = {
#     'random_state': 42,
#     'C': 0.01,
#     'solver': 'liblinear',
#     'penalty': 'l1',
#     'max_iter': 1000
# }

# # define model
# model = LogisticRegression(**config)
# # fit it
# model.fit(X_train_sm, y_train_sm)
# # test
# y_train_pred = model.predict(X_train_scaled)
# y_test_pred = model.predict(X_test_scaled)
# # performance
# evaluate(y_train, y_train_pred)
# evaluate(y_test, y_test_pred)

# ********************************************
# Random Forest - Laura Tender
# ********************************************
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.inspection import permutation_importance
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, plot_confusion_matrix, classification_report

# train_data = pd.read_csv("/content/processed_train.csv", delimiter = ',')
# test_data = pd.read_csv("/content/processed_test.csv", delimiter = ',')
# nr_rows = len(train_data)
# nr_rows_with_bigger_incomes = len(train_data[train_data['salary']=='>50K'])
# nr_rows_with_lower_incomes = len(train_data[train_data['salary']=='<=50K'])
# bigger_incomes_percent = round(nr_rows_with_bigger_incomes / nr_rows * 100, 2)
# lower_incomes_percent = round(nr_rows_with_lower_incomes / nr_rows * 100, 2)
# print(f'{lower_incomes_percent}% people win less that 50K while {bigger_incomes_percent}% more.')
# print(f'{nr_rows} training samples, {len(test_data)} test samples')

# print(f'Our proccesed data have {len(train_data.columns)} columns.')
# print(f'Columns: {train_data.columns}')

# columns_for_training = [column for column in train_data.columns if column.endswith("_cat")]
# columns_for_training.append("fnlwgt")
# len(columns_for_training)

# X_train = train_data[columns_for_training]
# y_train = train_data['salary_label']
# X_test = test_data[columns_for_training]
# y_test = test_data['salary_label']

# def train_random_forest(X_train, X_test, max_features):
#   Bag = RandomForestClassifier(n_estimators = 1000, max_features = max_features)
#   Bag = Bag.fit(X_train, y_train)

#   importances = Bag.feature_importances_
#   forest_importances = pd.Series(importances, index = columns_for_training).sort_values(ascending=False)

#   std = np.std([tree.feature_importances_ for tree in Bag.estimators_], axis = 0)

#   fig, ax = plt.subplots()
#   forest_importances.plot.bar(yerr = std, ax = ax)
#   ax.set_title("Feature importances using MDI")
#   ax.set_ylabel("Mean decrease in impurity")
#   fig.tight_layout()

#   print(forest_importances)

#   rf_test_pred = Bag.predict(X_test)
#   plot_confusion_matrix(Bag, X_test, y_test, cmap = plt.cm.PuBuGn)
#   print(classification_report(y_test, rf_test_pred))

#   return Bag, rf_test_pred

# feature_numbers = [13, 10, 7, 3]
# models = []
# predictions = []
# for feature_number in feature_numbers:
#   model, prediction = train_random_forest(X_train, X_test, feature_number)
#   models.append(model)
#   predictions.append(prediction)

# def compute_scores(models_names, predictions):
#   conclusion = pd.DataFrame({'Model': models_names})
#   accuracies, precisions, recalls, f1_scores  = [], [], [], []

#   for predicition in predictions:
#     accuracies.append(accuracy_score(y_test, predicition))
#     precisions.append(precision_score(y_test, predicition))
#     recalls.append(recall_score(y_test, predicition))
#     f1_scores.append(f1_score(y_test, predicition))

#   conclusion['accuracy'] = accuracies
#   conclusion['precision'] = precisions
#   conclusion['recall'] = recalls
#   conclusion['f1'] = f1_scores

#   return conclusion

# compute_scores(['Random Forest - Bagged','Random Forest - 10', 'Random Forest - 7', 'Random Forest - 3'], predictions)
# least_important_columns = ['sex_cat', 'marital-status_cat', 'capital-loss_cat']
# X_train.drop(columns=least_important_columns)
# X_test.drop(columns=least_important_columns)
# model, prediction = train_random_forest(X_train, X_test, 10)
# compute_scores(['Random Forest - Bagged - top 10 features'], [prediction])

# next_least_important_columns = ['race_cat', 'native-country_cat', 'hours-per-week_cat', 'capital-gain_cat']
# X_train.drop(columns=next_least_important_columns)
# X_test.drop(columns=next_least_important_columns)
# model, prediction = train_random_forest(X_train, X_test, 6)
# compute_scores(['Random Forest - Bagged - top 6 features'], [prediction])

# continuous_columns_for_training = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
# category_columns_for_training = ['workclass_cat', 'marital-status_cat', 'occupation_cat', 'relationship_cat', 'race_cat', 'sex_cat', 'native-country_cat']
# columns_for_training = continuous_columns_for_training + category_columns_for_training
# len(columns_for_training)

# least_important_columns = ['sex_cat', 'marital-status_cat', 'capital-loss_cat']
# X_train.drop(columns=least_important_columns)
# X_test.drop(columns=least_important_columns)
# model, prediction = train_random_forest(X_train, X_test, 10)
# compute_scores(['Random Forest - Bagged - top 10 features'], [prediction])

# next_least_important_columns = ['race_cat', 'native-country_cat', 'hours-per-week_cat', 'capital-gain_cat']
# X_train.drop(columns=next_least_important_columns)
# X_test.drop(columns=next_least_important_columns)
# model, prediction = train_random_forest(X_train, X_test, 6)
# compute_scores(['Random Forest - Bagged - top 6 features'], [prediction])

# continuous_columns_for_training = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
# category_columns_for_training = ['workclass_cat', 'marital-status_cat', 'occupation_cat', 'relationship_cat', 'race_cat', 'sex_cat', 'native-country_cat']
# columns_for_training = continuous_columns_for_training + category_columns_for_training
# len(columns_for_training)

# X_train = train_data[columns_for_training]
# X_test = test_data[columns_for_training]

# feature_numbers = [13, 9, 5]
# models = []
# predictions = []
# for feature_number in feature_numbers:
#   model, prediction = train_random_forest(X_train, X_test, feature_number)
#   models.append(model)
#   predictions.append(prediction)

# compute_scores(['Random Forest - Bagged','Random Forest - 9', 'Random Forest - 5'], predictions)

# least_important_columns = ['sex_cat', 'race_cat', 'native-country_cat', 'capital-loss', 'workclass_cat']
# X_train.drop(columns=least_important_columns)
# X_test.drop(columns=least_important_columns)
# model, prediction = train_random_forest(X_train, X_test, 8)
# compute_scores(['Random Forest - Bagged - 5 columns removed'], [prediction])

# model, prediction = train_random_forest(X_train, X_test, 5)

# compute_scores(['Random Forest - Bagged - 5 columns removed - 5 max_features'], [prediction])

# ********************************************
# SVM - Liviu Bouruc
# ********************************************
# #%%
# import numpy as np
# import pandas as pd 
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn import svm
# from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
# from sklearn.model_selection import cross_val_score, GridSearchCV
# from matplotlib import pyplot as plt
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, plot_confusion_matrix, classification_report
# from sklearn.preprocessing import scale

# #%%
# df = pd.read_csv('processed_train.csv')
# df.head()

# #%%
# data = df.iloc[:, :15]
# data = data.drop('education-num', axis=1)
# data = data.drop('fnlwgt', axis=1)
# data['salary'].replace(['<=50K', '>50K'], [0, 1], inplace=True)

# #%%
# labels = data['salary']
# data = data.drop('salary', axis=1)

# #%%
# continous_columns  = data[['age', 'capital-gain', 'capital-loss', 'hours-per-week']]
# categorical_columns = data[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']]

# encoded_columns = pd.get_dummies(categorical_columns)
# model_data = pd.concat([continous_columns, encoded_columns],axis=1)
# #%%
# model_data

# #%%
# train_data, val_data, train_labels, val_labels = train_test_split(model_data, labels)
# train_data = scale(train_data)
# val_data = scale(val_data)

# #%%
# for ker in ['rbf', 'linear']:
#     for c in [1, 1.5, 3]:
#         svm_classifier = svm.SVC(kernel=ker, C=c)
#         svm_classifier = svm_classifier.fit(train_data, train_labels)

#         print("------------------------------------")
#         print('kernel ' + ker + '; C: ' + str(c))
#         predictions = svm_classifier.predict(val_data)
#         print("Accuracy: " + str(accuracy_score(val_labels, predictions)) + "; F1: " + str(f1_score(val_labels, predictions)))
#         #if ker == 'linear':
#         #    pd.Series(abs(svm_classifier.coef_[0]), index=model_data.columns).nlargest(10).plot(kind='barh')
#         #plot_confusion_matrix(svm_classifier, val_data, val_labels, cmap = plt.cm.PuBuGn)

# #%%
# svm_classifier = svm.SVC(kernel='linear', C=1.5)
# svm_classifier = svm_classifier.fit(train_data, train_labels)

# #%%
# predictions = svm_classifier.predict(val_data)
# print(f1_score(val_labels, predictions))
# conf_mat = confusion_matrix(val_labels, predictions)
# conf_mat

# #%%
# pd.Series(abs(svm_classifier.coef_[0]), index=model_data.columns).nlargest(10).plot(kind='barh')

# #%%
# plot_confusion_matrix(svm_classifier, val_data, val_labels, cmap = plt.cm.PuBuGn)

# # %%
# print(recall_score(val_labels, predictions))


# #%%
# test_df = pd.read_csv('processed_test.csv')
# test_data = test_df.iloc[:, :15]
# test_data = test_data.drop('education-num', axis=1)
# test_data = test_data.drop('fnlwgt', axis=1)
# test_data['salary'].replace(['<=50K.', '>50K.'], [0, 1], inplace=True)

# test_labels = test_data['salary']
# test_data = test_data.drop('salary', axis=1)

# test_continous_columns  = test_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week']]
# test_categorical_columns = test_data[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']]
# test_encoded_columns = pd.get_dummies(test_categorical_columns)
# test_encoded_columns = test_encoded_columns.reindex(columns=encoded_columns.columns, fill_value=0)
# test_model_data = pd.concat([test_continous_columns, test_encoded_columns],axis=1)

# test_model_data = scale(test_model_data)

# test_predictions = svm_classifier.predict(test_model_data)
# print(f1_score(test_labels, test_predictions))
# conf_mat = confusion_matrix(test_labels, test_predictions)
# conf_mat

# #%%
# plot_confusion_matrix(svm_classifier, test_model_data, test_labels, cmap = plt.cm.PuBuGn)

# # %%
# print(recall_score(val_labels, predictions))

# ********************************************
# XGBoost - Darius Atudore
# ********************************************
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.inspection import permutation_importance
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# train_data = pd.read_csv("./data/processed_train.csv", delimiter = ',')
# train_data.head()
# test_data = pd.read_csv("./data/processed_test.csv", delimiter = ',')
# test_data.head()
# print(f'Our proccesed train data has {len(train_data.columns)} columns.')
# print(f'Columns: {train_data.columns}')

# def train_xgboost(X_train, y_train, X_val, y_val):
#     model = XGBClassifier()
#     model.fit(X_train, y_train)

#     # show importances
#     importances = model.feature_importances_
#     importances = pd.Series(importances, index = columns_for_training)
#     print("==== IMPORTANCES ====\n")
#     print(importances)

#     predictions = model.predict(X_val)
    
#     # plot confusion matrix
#     cm = confusion_matrix(y_val, predictions, labels=model.classes_)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
#     print("\n==== CONFUSION MATRIX ====")
#     disp.plot()
#     plt.show()
    
#     # show classification report
#     print("\n==== CLASSIFICATION REPORT ====\n")
#     print(classification_report(y_val, predictions))
    
#     return model, predictions
    
# def compute_result(y, predictions):   
#     acc = accuracy_score(y, predictions)
#     prec = precision_score(y, predictions)
#     rec = recall_score(y, predictions)
#     f1 = f1_score(y, predictions)

#     result = pd.DataFrame({'Model': ['XGBoost']})

#     result['accuracy'] = acc
#     result['precision'] = prec
#     result['recall'] = rec
#     result['f1'] = f1

#     return result

# def plot_features(model, X_val):
#     feature_importance = model.feature_importances_
#     sorted_idx = np.argsort(feature_importance)
#     fig = plt.figure(figsize=(12, 6))
#     plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
#     plt.yticks(range(len(sorted_idx)), np.array(X_val.columns)[sorted_idx])
#     plt.title('Feature Importance')

# continuous_columns_for_training = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
# category_columns_for_training = ['workclass_cat', 'marital-status_cat', 'occupation_cat', 'relationship_cat', 'race_cat', 'sex_cat', 'native-country_cat']
# columns_for_training = continuous_columns_for_training + category_columns_for_training
# len(columns_for_training)

# X_train = train_data[columns_for_training]
# y_train = train_data['salary_label']
# X_test = test_data[columns_for_training]
# y_test = test_data['salary_label']

# model_13, predictions_13 = train_xgboost(X_train, y_train, X_test, y_test)

# plot_features(model_13, X_test)
# result = compute_result(y_test, predictions_13)

# continuous_columns_for_training = ['age', 'education-num', 'capital-gain', 'hours-per-week', 'capital-loss']
# category_columns_for_training = ['marital-status_cat', 'occupation_cat', 'relationship_cat']
# columns_for_training = continuous_columns_for_training + category_columns_for_training
# len(columns_for_training)

# X_train = train_data[columns_for_training]
# y_train = train_data['salary_label']
# X_test = test_data[columns_for_training]
# y_test = test_data['salary_label']

# model_8, predictions_8 = train_xgboost(X_train, y_train, X_test, y_test)

# plot_features(model_8, X_test)
# result = compute_result(y_test, predictions_8)

# continuous_columns_for_training = ['education-num', 'capital-gain', 'capital-loss']
# category_columns_for_training = ['marital-status_cat', 'occupation_cat', 'relationship_cat']
# columns_for_training = continuous_columns_for_training + category_columns_for_training
# print(len(columns_for_training))

# X_train = train_data[columns_for_training]
# y_train = train_data['salary_label']
# X_test = test_data[columns_for_training]
# y_test = test_data['salary_label']

# model_6, predictions_6 = train_xgboost(X_train, y_train, X_test, y_test)

# plot_features(model_6, X_test)
# result = compute_result(y_test, predictions_6)
# test_data = pd.read_csv("./data/processed_test.csv", delimiter = ',')

# continuous_columns_for_training = ['age', 'education-num', 'capital-gain', 'hours-per-week', 'capital-loss']
# category_columns_for_training = ['marital-status_cat', 'occupation_cat', 'relationship_cat']
# columns_for_training = continuous_columns_for_training + category_columns_for_training
# len(columns_for_training)

# X = test_data[columns_for_training]
# y = test_data['salary_label']

# predictions_test = model_8.predict(X)
# result = compute_result(y, predictions_test)

# cm = confusion_matrix(y, predictions_test, labels=model_13.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_13.classes_)
# print("\n==== CONFUSION MATRIX ====")
# disp.plot()
# plt.show()