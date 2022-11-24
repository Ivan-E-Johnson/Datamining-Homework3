import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection import cross_val_score

def read_data(Filename):
	return 	pd.read_csv(Filename, sep= '\t', header= None)

def seperate_data(df):
	label = df.iloc[:,-1]
	df = df.iloc[:,:-1]
	return df, label

#SEE HOW DIFFERENT SCALERS AFFECT THE RESULTS OF THE CLASSIFIERS
def preprocess_data(df):
	minMax = preprocessing.RobustScaler()
	return pd.DataFrame(minMax.fit_transform(df.values))

def get_data():
	dataset1 = "hw3_dataset1.txt"
	dataset2 = "hw3_dataset2.txt"

	origdf1 = read_data(dataset1)
	origdf2 = read_data(dataset2)

	df1 , label1 = seperate_data(origdf1)
	df2 , label2 = seperate_data(origdf2)
	df2 = pd.get_dummies(df2)
	df1 = preprocess_data(df1)
	df2 = preprocess_data(df2)
	return df1, label1, df2, label2

def combine_df_label(df, label):
	return pd.concat([df,label], axis = 1)

def Generic(df,label, clf, test_size = .33):
	X_train , X_test , y_train , y_test = train_test_split(df , label , test_size=test_size)
	clf.fit(X_train , y_train)
	prob_predict = clf.predict_proba(X_test)
	predict = clf.predict(X_test)
	score = clf.score(X_test , y_test)

	return  X_test, y_test, predict, prob_predict, score, cross_val_score(clf,df, label, cv = 10 )

def KNN(df, label,nneighbors =5, test_size = .33):
	clf  = KNeighborsClassifier(nneighbors, weights= "uniform")
	return Generic(df, label, clf, test_size)

def NB(df, label,test_size = .33):
	clf  = GaussianNB()
	return Generic(df, label, clf, test_size)

def SVM(df, label,test_size = .33):
	clf  = SVC(probability=True)
	return Generic(df, label, clf, test_size)

def DT(df, label,test_size = .33, criterion ="gini",splitter = "best"):
	clf  = DecisionTreeClassifier(criterion=criterion,splitter=splitter)
	return Generic(df, label, clf, test_size)


def RandomForrest(df, label ,num_estimators = 100,test_size = .33 ,):
	clf = RandomForestClassifier(n_estimators = num_estimators, bootstrap=True)
	return Generic(df,label, clf, test_size)

def HistGradiantBoosted(df, label ,test_size = .33 ,):
	clf = HistGradientBoostingClassifier()
	return Generic(df,label, clf, test_size)










df1, label1, df2, label2 = get_data()

funcs = [KNN, NB, SVM, DT, RandomForrest, HistGradiantBoosted]
funcName = ["KNN", "NaiveBayes", "SupportVectorMachine", "DecisionTree", "RandomForrest", "HistGradiantBoosted"]
data = [df1, df2]
labels = [label1, label2]
dataName = ["data_set_1_", "data_set_2_"]
scores = {}
ten_fold_cross_validation_score = {}
for df,label, df_name in zip(data,labels,dataName):
	for func,name in zip(funcs, funcName):
		X_test, Y_test, prediction, prob_prediction, score, cross_val =  func(df,label)

		scores[df_name+name] = score
		ten_fold_cross_validation_score[df_name+name] = cross_val



# The number of features that can be searched at each split point (m) must be specified as a parameter to the algorithm. You can try different values and tune it using cross validation.
#
# For classification a good default is: m = sqrt(p)

#TODO implement RandomForrest based on Decision Tree
#TODO implement Booking based on implmentation of Decision Tree
#TODO Adopt  10-fold  Cross  Validation  to  evaluate  the  performance  of  all  methods  on  the
# provided two datasets in terms of Accuracy, Precision, Recall, and F-1 measure




print("Finished")




