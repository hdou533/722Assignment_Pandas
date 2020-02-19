# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 20:19:13 2019

@author: 35982
"""
#2. Data understanding
#2.1 Collect initial data
# Data are collected in Kaggle, with 70,000 rows and 13 columns
#2.2 Describe the data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
#import yellowbrick
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics

obj = pd.read_csv("D:\I3/cardio_objective.csv")
sub = pd.read_csv("D:\I3/cardio_subjective.csv")
exam = pd.read_csv("D:\I3/cardio_examination.csv")

print(obj.columns)
print(sub.columns)
print(exam.columns)

print(obj.shape)
print(sub.shape)
print(exam.shape)

# 2.3 Explore the data
#print a few rows to look at the data
print(obj.head())
print(sub.head())
print(exam.head())
#print the distribution of gender
print(obj["gender"].value_counts(dropna=False))
#a summary of each feature
pd.set_option('display.max_columns', None)
print(obj.describe())
print(sub.describe())
print(exam.describe())
#the distribution of cardio
print(exam["cardio"].value_counts(dropna=False))
#plt.hist(exam["cardio"])
#correlation between features
correlation_matrix = exam.corr()
plt.figure(figsize=(8,8))
ax = sns.heatmap(correlation_matrix, vmax=1, square=True, annot=True,fmt='.2f', cmap ='GnBu', cbar_kws={"shrink": .5}, robust=True)
ax.set(ylim=(0, 6))
plt.title('Correlation between the features in exam', fontsize=20)
plt.show()


#2.4 Verify the data quality
#check NULL values
#objective features
print(obj.count())
#subjective features
print(sub.count())
print(sub["smoke"].value_counts(dropna=False))
print(sub["alco"].value_counts(dropna=False))
print(sub["active"].value_counts(dropna=False))
#examination features
print(exam.count())
print(exam["cholesterol"].value_counts(dropna=False))

#3. Data preparation
#3.1 Select the data
#3.2 clean data
#objective features
# replace M to 1, F to 2
obj['gender'].replace('M','1',inplace = True)
obj['gender'].replace('F','2',inplace = True)
# fill the NAN with 2(female)
obj['gender'] = obj['gender'].fillna('2')
#to check if the strings and NAN in gender be replaced 
#print(obj["gender"].value_counts(dropna=False))
obj['height']=obj['height'].fillna(obj['height'].mean())
obj['weight']=obj['weight'].fillna(obj['weight'].mean())
# check if all the NULL values be cleaned in objective features
#print(obj.count())
#print(pd.isnull(obj).any())

# begin to clean the NULL in subjective features
sub['smoke'] = sub['smoke'].fillna(1.0)
sub['alco'] = sub['alco'].fillna(1.0)
sub['active'] = sub['active'].fillna(0.0)
#print(sub.count())
#print(pd.isnull(sub).any())
#print(sub["smoke"].value_counts(dropna=False))
#print(sub["alco"].value_counts(dropna=False))
#print(sub["active"].value_counts(dropna=False))

#clean data in examination features
exam['ap_hi']=exam['ap_hi'].fillna(exam['ap_hi'].mean())
exam['ap_lo']=exam['ap_lo'].fillna(exam['ap_lo'].mean())
exam['cholesterol']=exam['cholesterol'].fillna(2.0)
print(exam.count())
print(pd.isnull(exam).any())

#3.3 construct the data
#create a new column which strores age that recorded by year
obj['ageY'] = (obj['age']/365).round().astype('int')
#print(obj['ageY']) 

#3.4	Integrate various data sources
result = pd.merge(obj,sub,how ='left')
df = pd.merge(result, exam, how='left')
print(df.columns)
print(df.head())
print(df.shape)

#3.5 Format the data
df = df.astype(int)


#4. Data transformation
#4.1 Reduce the data
#Delete the column "id"&"age"
del df['id']
del df['age']
df.rename({'ageY':'age'}, axis=1, inplace=True)
print(df.columns)
print(df.shape) 

#4.2 project the data
#Down-sample majority class
from sklearn.utils import resample
#seperate observations into different dataframes
df_majority = df[df.cardio==0]
df_minority = df[df.cardio==1]
#Downsample majority class
df_majority_downsampled = resample(df_majority,
                             replace=False,
                             n_samples=33859,
                             random_state=123)
#combine minorty class with downsampled majority class
dfs = pd.concat([df_majority_downsampled, df_minority])
print(dfs.cardio.value_counts())


#5.	Data-mining method selection
#5.1 Match and discuss data mining methods
#5.2 Select the appropriate data-mining method


#6.	Data-mining algorithm selection
#6.1	Conduct exploratory analysis and discuss
# try out which one has the highes accuracy
# building train and test sample
sample = dfs.sample(frac = .20)
#print(sample.shape)
cols=[col for col in sample.columns if col not in ['cardio']]
X = sample[cols]
Y = sample['cardio']
#train data=70%,test=30% 
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=.3)

#building classification model
#set a dictinorary of classifiers
dict_classifiers = {
        "Nearest Neighbors": KNeighborsClassifier(), 
        "Linear SVM": SVC(kernel='linear', C=0.025),
        "RBF SVM": SVC(gamma=2, C=1),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Neural Net": MLPClassifier(alpha = 1),
        "AdaBoost": AdaBoostClassifier(),
        "Naive Bayes": GaussianNB(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "Logistic Regression": LogisticRegression()
    }

def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 5, verbose = True):
      
    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = time.time()
        classifier.fit(X_train, Y_train)
        t_end = time.time()
        
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)
        
        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score, 'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return dict_models
 
def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]
    
    display(df_.sort_values(by=sort_by, ascending=False))
    
dict_models = batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 8)
display_dict_models(dict_models)

#6.2	Select data-mining algorithms based on discussion
#6.3	Build/select appropriate model and choose relevant parameter
ada_params = {
    'n_estimators': [5,10,50,100, 500],
    'learning_rate':[0.5,0.8,1,1.2,1.5]
}

for n_est in ada_params['n_estimators']:
    for lr in ada_params['learning_rate']:
        clf = AdaBoostClassifier(n_estimators=n_est,
                                 learning_rate=lr)
        clf.fit(X_train, Y_train)
        train = clf.score(X_train, Y_train)
        test = clf.score(X_test, Y_test)
        print("For (n_estimator={}, learning_rate={}) train,test score: \t {:.5f}\t {:.5f} ".format(n_est, lr, train, test))
# Create adaboost classifer object
abc = AdaBoostClassifier()
# Train Adaboost Classifer
s_model = abc.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = s_model.predict(X_test)
# Model Accuracy
print("Accuracy(n_estimators=50, learning_rate=1):",metrics.accuracy_score(Y_test, Y_pred))

#7
#7.1 Create and justify test designs
#split the inputs and  the target 'cardio' 
cols=[col for col in dfs.columns if col not in ['cardio']]
print(cols)

data = dfs[cols]
target=dfs['cardio']
data_train, data_test, target_train, target_test=train_test_split(data,target,test_size=0.3)
#check the columns and numbers in training set
#print("inputs columns(training): ",data_train.columns)
#print(data_train.shape)
#check the columns and numbers in testing set
#print("inputs number(testing): ",data_test.columns)
#print(data_test.shape)
#check the target
#print(target.value_counts(dropna=False))
#print(target_train.value_counts(dropna=False))
#print(target_test.value_counts(dropna=False))

#7.2 Conduct data mining
# Create AdaBoost classifer object
ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.5)
# Train Adaboost Classifer
model = ada.fit(data_train, target_train)
#Predict the response for test dataset
target_pred = model.predict(data_test)
# Model Accuracy
print("Model Accuracy:",metrics.accuracy_score(target_test, target_pred))

#Interpret
#8.2 Visualize
#correlation between features
correlation_matrix = dfs.corr()
plt.figure(figsize=(11,11))
ax = sns.heatmap(correlation_matrix, vmax=1, square=True, annot=True,fmt='.2f', cmap ='GnBu', cbar_kws={"shrink": .5}, robust=True)
ax.set(ylim=(0, 12))
plt.title('Correlation matrix between the features', fontsize=20)
plt.show()

#correlation of cardio with other features
def display_corr_with_col(dfs, col):
    correlation_matrix = dfs.corr()
    correlation_type = correlation_matrix[col].copy()
    abs_correlation_type = correlation_type.apply(lambda x: abs(x))
    desc_corr_values = abs_correlation_type.sort_values(ascending=False)
    y_values = list(desc_corr_values.values)[1:]
    x_values = range(0,len(y_values))
    xlabels = list(desc_corr_values.keys())[1:]
    fig, ax = plt.subplots(figsize=(8,8))
    ax.bar(x_values, y_values)
    ax.set_title('The correlation of all features with {}'.format(col), fontsize=20)
    ax.set_ylabel('Pearson correlatie coefficient [abs waarde]', fontsize=16)
    plt.xticks(x_values, xlabels, rotation='vertical')
    plt.show()
 
display_corr_with_col(dfs, 'cardio')


#8.5 Iterate prior steps
#reduce unimportant features
del data['gender']
del data['height']
del data['smoke']
del data['alco']
del data['active']
del data['ap_hi']
del data['ap_lo']
del data['gluc']

ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.5)
# Train Adaboost Classifer
model = ada.fit(data_train, target_train)
#Predict the response for test dataset
target_pred = model.predict(data_test)
# Model Accuracy
print("Iterate Model Accuracy:",metrics.accuracy_score(target_test, target_pred))

