#%%
import numpy as np
import pandas as pd 
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, plot_confusion_matrix, classification_report
from sklearn.preprocessing import scale

#%%
df = pd.read_csv('processed_train.csv')
df.head()

#%%
data = df.iloc[:, :15]
data = data.drop('education-num', axis=1)
data = data.drop('fnlwgt', axis=1)
data['salary'].replace(['<=50K', '>50K'], [0, 1], inplace=True)

#%%
labels = data['salary']
data = data.drop('salary', axis=1)

#%%
continous_columns  = data[['age', 'capital-gain', 'capital-loss', 'hours-per-week']]
categorical_columns = data[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']]

encoded_columns = pd.get_dummies(categorical_columns)
model_data = pd.concat([continous_columns, encoded_columns],axis=1)
#%%
model_data

#%%
train_data, val_data, train_labels, val_labels = train_test_split(model_data, labels)
train_data = scale(train_data)
val_data = scale(val_data)

#%%
for ker in ['rbf', 'linear']:
    for c in [1, 1.5, 3]:
        svm_classifier = svm.SVC(kernel=ker, C=c)
        svm_classifier = svm_classifier.fit(train_data, train_labels)

        print("------------------------------------")
        print('kernel ' + ker + '; C: ' + str(c))
        predictions = svm_classifier.predict(val_data)
        print("Accuracy: " + str(accuracy_score(val_labels, predictions)) + "; F1: " + str(f1_score(val_labels, predictions)))
        #if ker == 'linear':
        #    pd.Series(abs(svm_classifier.coef_[0]), index=model_data.columns).nlargest(10).plot(kind='barh')
        #plot_confusion_matrix(svm_classifier, val_data, val_labels, cmap = plt.cm.PuBuGn)

#%%
svm_classifier = svm.SVC(kernel='linear', C=1.5)
svm_classifier = svm_classifier.fit(train_data, train_labels)

#%%
predictions = svm_classifier.predict(val_data)
print(f1_score(val_labels, predictions))
conf_mat = confusion_matrix(val_labels, predictions)
conf_mat

#%%
pd.Series(abs(svm_classifier.coef_[0]), index=model_data.columns).nlargest(10).plot(kind='barh')

#%%
plot_confusion_matrix(svm_classifier, val_data, val_labels, cmap = plt.cm.PuBuGn)

# %%
print(recall_score(val_labels, predictions))


#%%
test_df = pd.read_csv('processed_test.csv')
test_data = test_df.iloc[:, :15]
test_data = test_data.drop('education-num', axis=1)
test_data = test_data.drop('fnlwgt', axis=1)
test_data['salary'].replace(['<=50K.', '>50K.'], [0, 1], inplace=True)

test_labels = test_data['salary']
test_data = test_data.drop('salary', axis=1)

test_continous_columns  = test_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week']]
test_categorical_columns = test_data[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']]
test_encoded_columns = pd.get_dummies(test_categorical_columns)
test_encoded_columns = test_encoded_columns.reindex(columns=encoded_columns.columns, fill_value=0)
test_model_data = pd.concat([test_continous_columns, test_encoded_columns],axis=1)

test_model_data = scale(test_model_data)

test_predictions = svm_classifier.predict(test_model_data)
print(f1_score(test_labels, test_predictions))
conf_mat = confusion_matrix(test_labels, test_predictions)
conf_mat

#%%
plot_confusion_matrix(svm_classifier, test_model_data, test_labels, cmap = plt.cm.PuBuGn)

# %%
print(recall_score(val_labels, predictions))