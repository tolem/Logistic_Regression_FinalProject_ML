# project modules
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer  # import the KNNimputer class
# from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import binarize
from sklearn.metrics import roc_curve
matplotlib.use('TkAgg')

df = pd.read_csv('./data/data.csv')
df.drop(['RTD_ST_CD',], axis=1, inplace=True)
column_name = df.columns

# df.info()

categorical = [var for var in column_name if df[var].dtype == "O"]
numerical = [var for var in column_name if df[var].dtype != "O"]


# checks for missing vales in numerical and categorical
missing_obj = len([cols  for cols in categorical if df[cols].isnull().sum() > 0])
missing_num = len([cols  for cols in numerical if df[cols].isnull().sum() > 0])


def plot_missig_cols():  # shows a heatmap with missing cols
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.show()


# data visualization
# print(round(df['LOGINS'].describe(), 2))
# plt.figure(figsize=(30, 15))
# plt.subplot(2, 2, 1)
# fig = df.boxplot(column='Tenure')
# fig.set_title('Tenure')
# fig.set_ylabel('Tenure')
#
# plt.subplot(2, 2, 2)
# fig = df.boxplot(column='Age')
# fig.set_title('Age')
# fig.set_ylabel('Age')
#
# plt.subplot(2, 2, 3)
# fig = df.boxplot(column='LOGINS')
# df.boxplot(column='LOGINS')
# fig.set_title('Login')
# fig.set_ylabel('LOGINS')
# plt.show()
#
#
# plt.figure(figsize=(30, 15))
# plt.subplot(2, 2, 1)
# fig = df.boxplot(column='PAYMENTS_3M')
# fig.set_title('3 months Payment')
# fig.set_ylabel('PAYMENTS_3M')
#
# plt.subplot(2, 2, 2)
# fig = df.boxplot(column='PAYMENTS_6M')
# fig.set_title('6 months Payment')
# fig.set_ylabel('PAYMENTS_6M')
# plt.show()


# plt.figure(figsize=(15, 10))
# plt.subplot(2, 2, 1)
# fig = df.Tenure.hist(bins=10)
# fig.set_xlabel('Tenure')
# fig.set_ylabel('Call_Flag')
#
# plt.subplot(2, 2, 2)
# fig = df.Age.hist(bins=10)
# fig.set_xlabel('Age')
# fig.set_ylabel('Call_Flag')
#
# plt.subplot(2, 2, 3)
# fig = df.LOGINS.hist(bins=10)
# fig.set_xlabel('LOGINS')
# fig.set_ylabel('Call_Flag')
# plt.show()
#
#
# plt.figure(figsize=(15, 10))
# plt.subplot(2, 2, 1)
# fig = df.PAYMENTS_3M.hist(bins=10)
# fig.set_xlabel('PAYMENTS_3M')
# fig.set_ylabel('Call_Flag')
#
# plt.subplot(2, 2, 2)
# fig = df.PAYMENTS_6M.hist(bins=10)
# fig.set_xlabel('PAYMENTS_6M')
# fig.set_ylabel('Call_Flag')
# plt.show()

# IQR = df.Tenure.quantile(0.75) - df.Tenure.quantile(0.25)
# Lower_fence = df.Tenure.quantile(0.25) - (IQR * 3)
# Upper_fence = df.Tenure.quantile(0.75) + (IQR * 3)
# print('Tenure outliers are values < {lowerfence} or > {upperfence}.'.format(lowerfence=Lower_fence, upperfence=Upper_fence))
# #
# IQR = df.Age.quantile(0.75) - df.Age.quantile(0.25)
# Lower_fence = df.Age.quantile(0.25) - (IQR * 3)
# Upper_fence = df.Age.quantile(0.75) + (IQR * 3)
# print('Age outliers are values < {lowerfence} or > {upperfence}.'.format(lowerfence=Lower_fence, upperfence=Upper_fence))
# # quit()
# IQR = df.LOGINS.quantile(0.75) - df.LOGINS.quantile(0.25)
# Lower_fence = df.LOGINS.quantile(0.25) - (IQR * 3)
# Upper_fence = df.LOGINS.quantile(0.75) + (IQR * 3)
# print('LOGINS outliers are values < {lowerfence} or > {upperfence}.'.format(lowerfence=Lower_fence, upperfence=Upper_fence))
#
# IQR = df.PAYMENTS_3M.quantile(0.75) - df.PAYMENTS_3M.quantile(0.25)
# Lower_fence = df.PAYMENTS_3M.quantile(0.25) - (IQR * 3)
# Upper_fence = df.PAYMENTS_3M.quantile(0.75) + (IQR * 3)
# print('PAYMENTS_3M outliers are values < {lowerfence} or > {upperfence}.'.format(lowerfence=Lower_fence, upperfence=Upper_fence))
#
# IQR = df.PAYMENTS_6M.quantile(0.75) - df.PAYMENTS_6M.quantile(0.25)
# Lower_fence = df.PAYMENTS_6M.quantile(0.25) - (IQR * 3)
# Upper_fence = df.PAYMENTS_6M.quantile(0.75) + (IQR * 3)
# print('PAYMENTS_6M outliers are values < {lowerfence} or > {upperfence}.'.format(lowerfence=Lower_fence, upperfence=Upper_fence))
#
# exit()

# plot_missig_cols()
missing_col = [col for col in column_name if df[col].isnull().sum() > 0]


# shows class distribution with boxpolot
def customer_check(columns: 'outputs a 0 or num for all rows in a col') -> int:
    value = columns[0]
    if pd.isnull(value) or value == "NONE":
        return 0
    else:
        return int(value)


# encoding Cabin cols to handle missing cols
df['CustomerSegment'] = df[['CustomerSegment']].apply(customer_check, axis=1)

# round age columns
df['Age'] = round(df['Age'])


# parse date column
df['DATE_FOR'] = pd.to_datetime(df['DATE_FOR'])
df['Year'] = df['DATE_FOR'].dt.year
df['Month'] = df['DATE_FOR'].dt.month
df['Day'] = df['DATE_FOR'].dt.day




# plt.figure(figsize=(12, 7))
# sns.boxplot(x='Age', y='Tenure', data=df, palette='winter')
# sns.barplot(data=Counter(df['Call_Flag']), palette='winter')
# plt.show()

# plt.figure(figsize=(12, 7))
# sns.boxplot(x='Tenure', y='Age', data=df, palette='winter')
# plt.show()


# Press the green button in the gutter to run the script.
print(df.columns)
print(df.shape)

# print(df['GENDER'].nunique())
# print(df[categorical].isnull().sum())
# print(df[numerical].isnull().sum())


# missing_col.remove('RECENT_PAYMENT')

# standardize Tenure columns
df_x = df[['Age', ]]
df[['Age', ]] = (df_x - df_x.mean()) / df_x.std()

print(df[['Age', 'Tenure']].head())

# create an object for KNNImputer
imputer = KNNImputer(n_neighbors=2)
df[missing_col] = imputer.fit_transform(df[missing_col])


# print(df.isnull().sum())
print(missing_col)
print(df[['RECENT_PAYMENT']].isnull().sum())
print(df[['CustomerSegment']].isnull().sum())


# drop date column
df.drop('DATE_FOR', axis=1, inplace=True)
# one hot encoding for GENDER and Marital_Status
df[['GENDER_M', 'GENDER_F']] = pd.get_dummies(df.GENDER)
marital_columns = pd.get_dummies(df.MART_STATUS)

_marital = marital_columns.columns
df[_marital] = marital_columns

df.drop('GENDER', axis=1, inplace=True)
df.drop('MART_STATUS', axis=1, inplace=True)

X, y = df.drop('Call_Flag', axis=1), df['Call_Flag']
# imbalance datasets
# fig = sns.countplot(x='Call_Flag', data=df)
# fig.set_title('Imbalance Sample Size')
# plt.show()
# exit()

X.info()
# UnderSampling with tomek links
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)
nearest_neigh_removed = Counter(y)[0] - Counter(y_res)[0]
print('Before oversampling with Smote %s' % (Counter(y)),
      'After oversampling with Smote %s'  % Counter(y_res),
      sep='\n'
      )


# fig = sns.countplot(x=y_res, data=df, color="salmon", facecolor=(0, 0, 0, 0),
#                    linewidth=5,
#                    edgecolor=sns.color_palette("BrBG", 2))
# fig.set_title('Balance Sample Size with Smote')
# plt.show()
# exit()

X_train, X_test, y_train, y_test = \
    train_test_split(X_res,
                     y_res, test_size=0.2,
                     random_state=100)


# feature engineering
encoder = ce.BinaryEncoder(cols=['RECENT_PAYMENT', 'NOT_DI_3M', 'NOT_DI_6M', 'POLICYPURCHASECHANNEL',])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)


binary_encoded = ['RECENT_PAYMENT_0', 'RECENT_PAYMENT_1', 'NOT_DI_3M_0',
                  'NOT_DI_3M_1', 'NOT_DI_6M_0',
                  'NOT_DI_6M_1', 'POLICYPURCHASECHANNEL_0',
                  'POLICYPURCHASECHANNEL_1']

# numerical.remove('Age')
# print(X_test.columns)

_cols = [x for x in X_train.columns if x not in binary_encoded]
# print(_cols)


X_train = pd.concat([X_train[_cols], X_train[binary_encoded], ], axis=1)
X_test = pd.concat([X_test[_cols], X_test[binary_encoded], ], axis=1)

# print(X_train.columns)
X_train.info(), X_test.info()


def max_value(df3, variable, top):
    return np.where(df3[variable] > top, top, df3[variable])


for df3 in [X_train, X_test]:
    df3['Tenure'] = max_value(df3, 'Tenure', 36.69)
    df3['PAYMENTS_3M'] = max_value(df3, 'PAYMENTS_3M', 6.0)
    df3['PAYMENTS_6M'] = max_value(df3, 'PAYMENTS_6M', 12.0)
    df3['LOGINS'] = max_value(df3, 'LOGINS', 4.0)


# print(X_train.shape, X_test.shape)
# print(X_train[''].columns)
# print(X_train['NOT_DI_6M_1'].head())
# print(Counter(y_test), Counter(y_train))
# print(y_test.shape, y_train.shape)


cols = X_train.columns
# scaler instance
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)  # fit train with scaler
X_test = scaler.fit_transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])



# Logistic Regression
logmodel = LogisticRegression(solver='liblinear', random_state=0)

logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)  # gets prediction
# outputs prediction
print(classification_report(y_test, predictions))
print("Accuracy train:", accuracy_score(y_test, predictions))
test_predictions = logmodel.predict(X_train)
print("Accuracy test:", accuracy_score(y_train, test_predictions))



print(type(y_test), type(predictions))

# model
# print('Model training accuracy score: {0:0.10f}'.format(accuracy_score(y_train, predictions)))
# print('Model testing accuracy score: {0:0.10f}'.format(accuracy_score(y_test, test_predictions)))


# nulls accuracy
# print(y_test.value_counts())
# null_accuracy = (25069) / (25069 + 25060)
# print('null accuracy is '+ str(null_accuracy))
#
# # confusion_matrix
# cm = confusion_matrix(y_test, predictions)
# print('Confusion matrix\n', cm)
# print('True Positives (TP) = ', cm[0, 0])
# print('True Negatives (TN) = ', cm[1, 1])
# print('False Positives (FP) = ', cm[0, 1])
# print('False Negatives (FN) = ', cm[1, 0])
#


# cm_matrix = pd.DataFrame(data=cm,
#                          columns=['Actual Positive', 'Actual Negative'],
#                          index=['Predict Positive', 'Predict Negative'])
#
# sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
# plt.show()


# TP = cm[0, 0]
# TN = cm[1, 1]
# FP = cm[0, 1]
# FN = cm[1, 0]

# print(X_test.shape, X_train.shape)
# classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
# print('classification accuracy: {0:0.2f}'.format(classification_accuracy))
# classification_error = (FP + FN) / float(TP + TN + FP + FN)
# print('classification error: {0:0.2f}'.format(classification_error))


y_pred1 = logmodel.predict_proba(X_test)[:, 1]
print(type(y_test), type(y_pred1))
print(y_pred1[:5])
y_pred1 = logmodel.predict_proba(X_test)[:, 1]
y_pred1 = y_pred1.reshape(-1, 1)
y_pred2 = binarize(y_pred1, threshold= 4 / 10)
y_pred2 = np.where(y_pred2 == 1, 'Yes', 'No')
y_test_arr = y_test.to_numpy()
# print(y_pred2[:5])
print(y_test_arr[:5])
y_test_arr = np.where(y_test_arr == 1, 'Yes', 'No')
print(y_test_arr[:5])

for i in range(1, 6):
    cm1 = 0
    y_pred1 = logmodel.predict_proba(X_test)[:, 1]
    y_pred1 = y_pred1.reshape(-1, 1)
    y_pred2 = binarize(y_pred1, threshold=i / 10)
    # y_pred2 = np.where(y_pred2 == 1, 'Yes', 'No')
    cm1 = confusion_matrix(y_test, y_pred2)
    print('With', i / 10, 'threshold the Confusion Matrix is ', '\n\n', cm1, '\n\n',
          'with', cm1[0, 0] + cm1[1, 1], 'correct predictions, ', '\n\n', cm1[0, 1],
          'Type I errors ( Fasle Positive), ', '\n\n',
          cm1[1, 0], 'Type II errors ( False Negative), ', '\n\n',
          'Accuracy score: ', (accuracy_score(y_test, y_pred2)),
          '\n\n',
          'Sensitivity: ', cm1[1, 1] / float(cm1[1, 1] + cm1[1, 0]), '\n\n',
          'Specificity: ', cm1[0, 0] / float(cm1[0, 0] + cm1[0, 1]), '\n\n',
          '==================================================', '\n\n')


# y_pred1 = logmodel.predict_proba(X_test)[:, 1]
# print(y_test.shape, type(y_test.to_numpy()), type(y_pred1))
# fpr, tpr, thresholds = roc_curve(y_test.to_numpy(), y_pred1)
# plt.figure(figsize=(6, 4))
# plt.plot(fpr, tpr, linewidth=2)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.rcParams['font.size'] = 12
# plt.title('ROC curve for Call_Flag classifier')
# plt.xlabel('False Positive Rate (1 - Specificity)')
# plt.ylabel('True Positive Rate (Sensitivity)')
# plt.show()


# scores = cross_val_score(logmodel, X_train, y_train, cv=5, scoring='accuracy')
# print('Cross-validation score: {}'.format(scores))
# print('Average cross-validation score:{:.4f}'.format(scores.mean()))
#
# parameters = [{'penalty': ['l1', 'l2']}, {'C': [1, 10, 100, 1000]}]
# grid_search = GridSearchCV(estimator=logmodel, param_grid=parameters, scoring='accuracy', cv=5, verbose=0)
# grid_search.fit(X_train, y_train)
# print(grid_search)
# print('GridSearch CV best score: {:.4f}'.format(grid_search.best_score_))
# print('Parameters that give the best results:', grid_search.best_params_)
# print('Estimator that was chosen by the search:', grid_search.best_estimator_)
# print('GridSearch CV score on the test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))




























