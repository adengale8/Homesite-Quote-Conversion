#Import Python Packages
#from google.colab import drive
#drive.mount('/content/drive/')
from google.colab import drive
drive.mount('/gdrive')

!pip install vecstack

#Import all necessary library
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, precision_score, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import DataConversionWarning
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")
from vecstack import stacking
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

#Read training data file
trainfile = r'/gdrive/MyDrive/ColabNotebooks/HW3Train.csv'
trainData = pd.read_csv(trainfile)

#Read test data file
testfile = r'/gdrive/MyDrive/ColabNotebooks/HW3Test.csv'
testData = pd.read_csv(testfile)

trainData.head()
#print("=======")
#testData.head()

#To get list of names of all Columns from a dataframe

TrainCols = list(trainData.columns.values)
TestCols = list(testData.columns.values)
print(TrainCols)
print(TestCols)

# Impute the missing values in the data set
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = imputer.fit_transform(trainData)

imputer1 = SimpleImputer(strategy='most_frequent')
df_imputed = imputer.fit_transform(testData)

testData.isnull().sum().sort_values(ascending=False)

# Assuming you've already loaded your datasets into train_data and test_data
# Calculate variance of each feature
variance = trainData[TrainCols[0:len(TrainCols)-1]].var().sort_values(ascending=False)

# Select the names of the top 100 columns with highest variance
top_100_cols = variance.head(100).index.tolist()

# Filter the datasets to include only these top 100 columns
train_data_top_100 = trainData[top_100_cols]
test_data_top_100 = testData[top_100_cols]

# Print the shapes to verify
print(train_data_top_100.shape)
print(test_data_top_100.shape)

# Seperate Target column from Train Data
Xtrain = train_data_top_100
Ytrain = trainData[['QuoteConversion_Flag']]
print(Xtrain.shape)
print(Ytrain.shape)
Xtest = test_data_top_100
print(Xtest.shape)
print(Ytrain.value_counts())

X_train1, X_test, Y_train1, Y_test = train_test_split(Xtrain, Ytrain, test_size = .20, random_state = 1)
smote = SMOTE(random_state = 20, sampling_strategy=0.5)
X_train, Y_train = smote.fit_resample(X_train1, Y_train1)
print(Y_train.value_counts())

print("___________________________________________________________________________________________\nEnsemble Methods Predictions using GradientBoosting, RandomForest and Decision Tree Classifier\n")

models = [ KNeighborsClassifier(), MLPClassifier(), LinearSVC(), RandomForestClassifier(), DecisionTreeClassifier() ]

S_Train, S_Test = stacking(models,
                           X_train, Y_train, X_test,
                           regression=False,

                           mode='oof_pred_bag',

                           needs_proba=False,

                           save_dir=None,

                           metric=accuracy_score,

                           n_folds=4,

                           stratified=True,

                           shuffle=True,

                           random_state=0,

                           verbose=2)

mlp = MLPClassifier(random_state=42)

# Define hyperparameters to search
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(mlp, param_grid, cv=3, n_jobs=-1)
grid_search.fit(S_Train, Y_train)

# Get the best hyperparameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(best_params)
print(best_score)

best_mlp = MLPClassifier(**best_params)
best_mlp.fit(S_Train, Y_train)
Y_pred_mlp = best_mlp.predict(S_Test)
print(classification_report(Y_test, Y_pred_mlp))
print(accuracy_score(Y_test, Y_pred_mlp))
print(roc_auc_score(Y_test, Y_pred_mlp))
clf_cv_score = cross_val_score(best_mlp, S_Test, Y_test, cv=10, scoring="roc_auc")
print(clf_cv_score.mean())

print("___________________________________________________________________________________________\nEnsemble Methods Predictions using KNeighborsClassifier, MLPClassifier, LinearSVC, RandomForest and Decision Tree Classifier\n")

models_final = [ KNeighborsClassifier(), MLPClassifier(), LinearSVC(), RandomForestClassifier(), DecisionTreeClassifier() ]

S_Train_f, S_Test_f = stacking(models_final,
                           X_train, Y_train, Xtest,
                           regression=False,

                           mode='oof_pred_bag',

                           needs_proba=False,

                           save_dir=None,

                           metric=accuracy_score,

                           n_folds=4,

                           stratified=True,

                           shuffle=True,

                           random_state=0,

                           verbose=2)

mlp_f = MLPClassifier(random_state=42)

# Define hyperparameters to search
param_grid_f = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
}

# Perform GridSearchCV for hyperparameter tuning
random_search_f = RandomizedSearchCV(estimator=mlp_f, param_distributions=param_grid_f,
                                   n_iter=15, cv=5, random_state=42)
random_search_f.fit(S_Train_f, Y_train)

# Get the best hyperparameters and score
best_params = random_search_f.best_params_
best_score = random_search_f.best_score_
print(best_params)
print(best_score)

best_mlp_f = MLPClassifier(**best_params)
best_mlp_f.fit(S_Train_f, Y_train)
Y_pred_mlp_f = best_mlp_f.predict(S_Test_f)
wkk = pd.DataFrame({'QuoteNumber': testData['QuoteNumber'], 'QuoteConversion_Flag': Y_pred_mlp_f})
wkk.to_csv('/content/ML/model5.csv', index=False)

"""# **New SMOTE - 0.75**"""

X_train2, X_test, Y_train2, Y_test = train_test_split(Xtrain, Ytrain, test_size = .20, random_state = 1)
smote = SMOTE(random_state = 20, sampling_strategy=0.75)
X_train_s, Y_train_s = smote.fit_resample(X_train2, Y_train2)
print(Y_train_s.value_counts())

print("___________________________________________________________________________________________\nEnsemble Methods Predictions using KNeighborsClassifier, MLPClassifier, LinearSVC, RandomForest and Decision Tree Classifier\n")

models_s = [ KNeighborsClassifier(), MLPClassifier(), LinearSVC(), RandomForestClassifier(), DecisionTreeClassifier() ]

S_Train_s, S_Test_s = stacking(models_s,
                           X_train_s, Y_train_s, X_test,
                           regression=False,

                           mode='oof_pred_bag',

                           needs_proba=False,

                           save_dir=None,

                           metric=accuracy_score,

                           n_folds=4,

                           stratified=True,

                           shuffle=True,

                           random_state=0,

                           verbose=2)

mlp1 = MLPClassifier(random_state=42)

# Define hyperparameters to search
param_grid1 = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
}

# Perform GridSearchCV for hyperparameter tuning
grid_search_1 = GridSearchCV(mlp1, param_grid1, cv=3, n_jobs=-1)
grid_search_1.fit(S_Train_s, Y_train_s)

# Get the best hyperparameters and score
best_params1 = grid_search.best_params_
best_score1 = grid_search.best_score_
print(best_params1)
print(best_score1)

best_mlp1 = MLPClassifier(**best_params1)
best_mlp1.fit(S_Train_s, Y_train_s)
Y_pred_mlp1 = best_mlp1.predict(S_Test_s)
print(classification_report(Y_test, Y_pred_mlp1))
print(accuracy_score(Y_test, Y_pred_mlp1))
print(roc_auc_score(Y_test, Y_pred_mlp1))
clf_cv_score = cross_val_score(best_mlp, S_Test_s, Y_test, cv=10, scoring="roc_auc")
print(clf_cv_score.mean())

print("___________________________________________________________________________________________\nEnsemble Methods Predictions using KNeighborsClassifier, MLPClassifier, LinearSVC, RandomForest and Decision Tree Classifier\n")

models_s_f = [ KNeighborsClassifier(), MLPClassifier(), LinearSVC(), RandomForestClassifier(), DecisionTreeClassifier() ]

S_Train_s_f, S_Test_s_f = stacking(models_s_f,
                           X_train_s, Y_train_s, Xtest,
                           regression=False,

                           mode='oof_pred_bag',

                           needs_proba=False,

                           save_dir=None,

                           metric=accuracy_score,

                           n_folds=4,

                           stratified=True,

                           shuffle=True,

                           random_state=0,

                           verbose=2)

mlp1_f = MLPClassifier(random_state=42)

# Define hyperparameters to search
param_grid1_f = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
}

# Perform GridSearchCV for hyperparameter tuning
random_search1_f = RandomizedSearchCV(estimator= mlp1_f, param_distributions=param_grid_f,
                                   n_iter=15, cv=5, random_state=42)
random_search1_f.fit(S_Train_s_f, Y_train_s)

# Get the best hyperparameters and score
best_params1_f = random_search1_f.best_params_
best_score1_f = random_search1_f.best_score_
print(best_params1_f)
print(best_score1_f)

best_mlp1_f = MLPClassifier(**best_params1_f)
best_mlp1_f.fit(S_Train_s_f, Y_train_s)
Y_pred_mlp1 = best_mlp1_f.predict(S_Test_s_f)
wkk1 = pd.DataFrame({'QuoteNumber': testData['QuoteNumber'], 'QuoteConversion_Flag': Y_pred_mlp_f})
wkk1.to_csv('/content/ML/model6.csv', index=False)

"""# **New SMOTE - 1.0**"""

X_train3, X_test, Y_train3, Y_test = train_test_split(Xtrain, Ytrain, test_size = .20, random_state = 1)
smote = SMOTE(random_state = 20)
X_train_s1, Y_train_s1 = smote.fit_resample(X_train3, Y_train3)
print(Y_train_s1.value_counts())

print("___________________________________________________________________________________________\nEnsemble Methods Predictions using KNeighborsClassifier, MLPClassifier, LinearSVC, RandomForest and Decision Tree Classifier\n")

models_s1 = [ KNeighborsClassifier(), MLPClassifier(), LinearSVC(), RandomForestClassifier(), DecisionTreeClassifier() ]

S_Train_s1, S_Test_s1 = stacking(models_s1,
                           X_train_s1, Y_train_s1, X_test,
                           regression=False,

                           mode='oof_pred_bag',

                           needs_proba=False,

                           save_dir=None,

                           metric=accuracy_score,

                           n_folds=4,

                           stratified=True,

                           shuffle=True,

                           random_state=0,

                           verbose=2)

mlp2 = MLPClassifier(random_state=42)

# Define hyperparameters to search
param_grid2 = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
}

# Perform GridSearchCV for hyperparameter tuning
grid_search_2 = GridSearchCV(mlp2, param_grid2, cv=3, n_jobs=-1)
grid_search_2.fit(S_Train_s1, Y_train_s1)

# Get the best hyperparameters and score
best_params2 = grid_search.best_params_
best_score2 = grid_search.best_score_
print(best_params2)
print(best_score2)

best_mlp2 = MLPClassifier(**best_params2)
best_mlp2.fit(S_Train_s1, Y_train_s1)
Y_pred_mlp2 = best_mlp2.predict(S_Test_s1)
print(classification_report(Y_test, Y_pred_mlp2))
print(accuracy_score(Y_test, Y_pred_mlp2))
print(roc_auc_score(Y_test, Y_pred_mlp2))
clf_cv_score = cross_val_score(best_mlp2, S_Test_s1, Y_test, cv=10, scoring="roc_auc")
print(clf_cv_score.mean())

print("___________________________________________________________________________________________\nEnsemble Methods Predictions using KNeighborsClassifier, MLPClassifier, LinearSVC, RandomForest and Decision Tree Classifier\n")

models_s1_f = [ KNeighborsClassifier(), MLPClassifier(), LinearSVC(), RandomForestClassifier(), DecisionTreeClassifier() ]

S_Train_s1_f, S_Test_s1_f = stacking(models_s1_f,
                           X_train_s1, Y_train_s1, Xtest,
                           regression=False,

                           mode='oof_pred_bag',

                           needs_proba=False,

                           save_dir=None,

                           metric=accuracy_score,

                           n_folds=4,

                           stratified=True,

                           shuffle=True,

                           random_state=0,

                           verbose=2)

mlp2_f = MLPClassifier(random_state=42)

# Define hyperparameters to search
param_grid2_f = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
}

# Perform GridSearchCV for hyperparameter tuning
random_search1_f = RandomizedSearchCV(estimator= mlp2_f, param_distributions=param_grid2_f,
                                   n_iter=15, cv=5, random_state=42)
random_search1_f.fit(S_Train_s1_f, Y_train_s1)

# Get the best hyperparameters and score
best_params2_f = grid_search.best_params_
best_score2_f = grid_search.best_score_
print(best_params2_f)
print(best_score2_f)

best_mlp2_f = MLPClassifier(**best_params2_f)
best_mlp2_f.fit(S_Train_s1_f, Y_train_s1)
Y_pred_mlp2_f = best_mlp2_f.predict(S_Test_s1_f)

test_pred = best_mlp.predict(Xtest)
wkk = pd.DataFrame({'QuoteNumber': testData['QuoteNumber'], 'TARGET': test_pred})
wkk.to_csv('/content/ML/model4.csv', index=False)

test_pred1 = best_mlp1.predict(Xtest)
wkk1 = pd.DataFrame({'QuoteNumber': testData['QuoteNumber'], 'QuoteConversion_Flag': test_pred})
wkk1.to_csv('/content/ML/model5.csv', index=False)

test_pred2 = best_mlp2.predict(Xtest)
wkk2 = pd.DataFrame({'QuoteNumber': testData['QuoteNumber'], 'QuoteConversion_Flag': test_pred})
wkk2.to_csv('/content/ML/model6.csv', index=False)
