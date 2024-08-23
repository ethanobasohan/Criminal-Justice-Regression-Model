#@title Run this to import libraries and load data
!wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Projects%20-%20AI%20and%20Ethics%20-%20Criminal%20Justice/compas-scores-two-years.csv"

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay


data = pd.read_csv("compas-scores-two-years.csv", header=0)
data.head(n=20)


df = data.drop(labels=['id', 'name', 'first', 'last', 'compas_screening_date', 'dob', 'days_b_screening_arrest',
                         'c_jail_in', 'c_jail_out', 'c_case_number', 'c_offense_date', 'c_arrest_date', 'c_days_from_compas',
                         'r_case_number', 'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc',
                         'r_jail_in', 'r_jail_out', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date', 'decile_score.1',
                         'violent_recid', 'vr_charge_desc', 'in_custody', 'out_custody', 'priors_count.1', 'start', 'end',
                         'v_screening_date', 'event', 'type_of_assessment', 'v_type_of_assessment', 'screening_date',
                         'score_text', 'v_score_text', 'v_decile_score', 'decile_score', 'is_recid', 'is_violent_recid'], axis=1)
df.columns = ['sex', 'age', 'age_category', 'race', 'juvenile_felony_count', 'juvenile_misdemeanor_count', 'juvenile_other_count',
              'prior_convictions', 'current_charge', 'charge_description', 'recidivated_last_two_years']
df.head()


value_counts = df['charge_description'].value_counts()
df = df[df['charge_description'].isin(value_counts[value_counts >= 70].index)].reset_index(drop=True) # drop rare charges
for colname in df.select_dtypes(include='object').columns: # use get_dummies repeatedly one-hot encode categorical columns
  one_hot = pd.get_dummies(df[colname])
  df = df.drop(colname, axis=1)
  df = df.join(one_hot)
df


y_column = 'recidivated_last_two_years'
X_all, y_all = df.drop(y_column, axis=1), df[y_column]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3)

X_caucasian = X_test[X_test['Caucasian'] == 1]
y_caucasian = y_test[X_test['Caucasian'] == 1]
X_african_american = X_test[X_test['African-American'] == 1]
y_african_american = y_test[X_test['African-American'] == 1]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Training accuracy:", model.score(X_train, y_train))
print("Testing accuracy:", model.score(X_test, y_test))

# Question 2.5
from sklearn import svm
from sklearn.svm import LinearSVC
pyplot.rcParams['figure.figsize'] = [15, 10]


#Create and train the SVM model
model_svm = svm.SVC(kernel='linear')
model_svm.fit(X_train, y_train)

#Print the training and test accuracies
print("SVM Training accuracy:", model_svm.score(X_train, y_train))
print("SVM Testing accuracy:", model_svm.score(X_test, y_test))



# Question 2.6.
#Get the model coefficients (feature importances)
importance_svm = model_svm.coef_[0]

#Plot the feature importances
features = X_all.columns
pyplot.xticks(rotation="vertical")
pyplot.gca().tick_params(axis='both', which='major', labelsize=20)

svm_importance_plot = pyplot.bar(features, importance_svm)
pyplot.xlabel("Feature", fontsize=20)
pyplot.ylabel("Coefficient Value", fontsize=20)
pyplot.show()

# Question 2.7.
from sklearn.ensemble import RandomForestClassifier

#Create and train the random forest classifier
model_rf = RandomForestClassifier(max_depth=____) # fill this in
model_rf.fit(X_train, y_train)

#Print training and test accuracies
print("Random Forest Training accuracy:", model_rf.score(____, ____)) # fill this in
print("Random Forest Testing accuracy:", model_rf.score(____, ____)) # fill this in

# Question 2.8.
#Get feature importances
rf_importances = model_rf.feature_importances_

#Plot feature importances
pyplot.xticks(rotation="vertical")
pyplot.gca().tick_params(axis='both', which='major', labelsize=20)
rf_importance_plot = pyplot.bar(features, rf_importances)
pyplot.xlabel("Feature", fontsize=20)
pyplot.ylabel("Coefficient Value", fontsize=20)
pyplot.show()

#Create a list of importances associated with only the race-related variables
race_importances = rf_importances[10:16]
race_features = features[10:16]

#Plot feature importances related to race
pyplot.gca().tick_params(axis='both', which='major', labelsize=18)
race_importances_plot = pyplot.bar(race_features, race_importances)
pyplot.xlabel("Feature", fontsize=20)
pyplot.ylabel("Coefficient Value", fontsize=20)
pyplot.show()

# Question 2.9.
from sklearn.neural_network import MLPClassifier

#Create and train the neural netowrk
model_nn = MLPClassifier(hidden_layer_sizes=(____, ____, ____),random_state=1, max_iter=500) # fill this line in!
model_nn.fit(X_train, y_train)

#Print training and test accuracies
print("Neural Network Training accuracy:", model_nn.score(X_train, y_train))
print("Neural Network Testing accuracy:", model_nn.score(X_test, y_test))
