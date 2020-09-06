import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle

# Open data

data = pd.read_excel('DATA E.COLI ENGLISHMAN RIVER.xlsx')
X = data.loc[:, ~data.columns.isin(['E. Coli', 'Season', 'Threshold'])]
Y = data.loc[:, ['E. Coli']]

# ___Convert to numpy array___
X.to_numpy(dtype=object)
Y.to_numpy(dtype=object)

# __Generate train, test, and validation set__

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)
X_train, X_validate, Y_train, Y_validate = train_test_split(
    X_train, Y_train, test_size=0.25, random_state=0)

# ___Convert to zeros and ones with threshold 20___

Y_threshold_train = np.zeros_like(Y_train)
Y_threshold_train[Y_train >= 20] = 1
Y_threshold_validate = np.zeros_like(Y_validate)
Y_threshold_validate[Y_validate >= 20] = 1
Y_threshold_test = np.zeros_like(Y_test)
Y_threshold_test[Y_test >= 20] = 1

# __Verification of class balance in the train, test, validation sets__

plt.hist(Y_threshold_train)
plt.title('Histogram train data', size=18)
plt.show()
plt.hist(Y_threshold_validate)
plt.title('Histogram validation data', size=18)
plt.show()
plt.hist(Y_threshold_test)
plt.title('Histogram test data', size=18)
plt.show()

# __Compute mean and standard deviation of train data for normalization__

stdscX = StandardScaler()
stdscX.fit(X_train)

stdscY = StandardScaler()
stdscY.fit(Y_train)

# __Normalization of Train, Test, and validation Sets__

X_train = stdscX.transform(X_train)
X_test = stdscX.transform(X_test)
X_validate = stdscX.transform(X_validate)

Y_train = stdscY.transform(Y_train)

# __Plot the covariance matrix___

combined_data = np.hstack((X_train, Y_train))

# __Name the columns__
names = ['Hardness', 'Conductivity', 'Water temperature', 'Turbidity', 'Air temperature', 'Precipitation today',
         'Precipitation last 3 days', 'E. Coli']

cov_mat = np.cov(combined_data.T)
plt.figure(figsize=(10, 10))
plt.title('Correlation matrix Englishman River', size=18)
ax = sns.heatmap(
    cov_mat,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    cbar=True,
    square=True,
    yticklabels=names,
    xticklabels=names,
)
plt.tight_layout()
plt.show()

# Training Models

# 1. Logistic regression

t0 = time.time()
lab_enc = preprocessing.LabelEncoder()
Y_train_encoded = lab_enc.fit_transform(Y_threshold_train)
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X_train, Y_train_encoded)
delta = time.time() - t0
print(f'{"Training Time Logistic regression"}: {delta}s')

# __Generate report of Logistic regression__

# __Validate__

lab_enc = preprocessing.LabelEncoder()
Y_validate_encoded = lab_enc.fit_transform(Y_threshold_validate)
predictions_logistic = clf.predict(X_validate)
print(confusion_matrix(Y_validate_encoded, predictions_logistic))
print(classification_report(Y_validate_encoded, predictions_logistic))

# __Test__

lab_enc = preprocessing.LabelEncoder()
Y_test_encoded = lab_enc.fit_transform(Y_threshold_test)
predictions_logistic_test = clf.predict(X_test)
print(confusion_matrix(Y_test_encoded, predictions_logistic_test))
print(classification_report(Y_test_encoded, predictions_logistic_test))

# 2.Random forest Classifier

t0 = time.time()
forestClassifier = RandomForestClassifier(n_estimators=2000, random_state=0)
forestClassifier.fit(X_train, Y_train_encoded)
delta_learning_RF = time.time() - t0
print(f'{"Training Time with Random Forest classifier"}: {delta_learning_RF}s')

# __Generate report of Random Forest__

# __Validate__

t0 = time.time()
predictions_RF_validate = forestClassifier.predict(X_validate)
delta_predict_RF = time.time() - t0
print(f'{"Predicting Time with Random Forest classifier"}: {delta_predict_RF}s')
print(confusion_matrix(Y_validate_encoded, predictions_RF_validate))
print(classification_report(Y_validate_encoded, predictions_RF_validate))

# __Test__

t0 = time.time()
predictions_RF_test = forestClassifier.predict(X_test)
delta_predict_RF = time.time() - t0
print(f'{"Predicting Time with Random Forest classifier"}: {delta_predict_RF}s')
print(confusion_matrix(Y_test_encoded, predictions_RF_test))
print(classification_report(Y_test_encoded, predictions_RF_test))


# 3. Neural Network Classifier

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=10000)
t0 = time.time()
mlp.fit(X_train, Y_train_encoded)
delta_NN = time.time() - t0
print(f'{"Training Time with Neural Network classifier"}: {delta_NN}s')

# __Generate report of Neural Networks__

# __Validate__

t0 = time.time()
predictions_NN_validate = mlp.predict(X_validate)
delta_predict_NN = time.time() - t0
print(f'{"Prediction Time with Neural Network classifier"}: {delta_predict_NN}s')
print(confusion_matrix(Y_validate_encoded, predictions_NN_validate))
print(classification_report(Y_validate_encoded, predictions_NN_validate))

# __Test__

t0 = time.time()
predictions_NN_test = mlp.predict(X_test)
delta_predict_NN = time.time() - t0
print(f'{"Prediction Time with Neural Network classifier"}: {delta_predict_NN}s')
print(confusion_matrix(Y_test_encoded, predictions_NN_test))
print(classification_report(Y_test_encoded, predictions_NN_test))

# __Save__

my_model =	{
  "Model": clf,
  "Scaler": stdscX,
}

filename = 'finalized_model.sav'
pickle.dump(my_model, open(filename, 'wb'))