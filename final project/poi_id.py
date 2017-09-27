#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import scipy
import pandas
import numpy
import tester
import matplotlib
from matplotlib import pyplot
from sklearn.feature_selection import SelectKBest
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary',
                 'bonus',
  'deferred_income',
  'director_fees',
  'exercised_stock_options',
  'expenses',
  'loan_advances',
  'long_term_incentive',
  'restricted_stock',
  'restricted_stock_deferred',
  'shared_receipt_with_poi',
  'total_payments',
  'total_stock_value',
'from_messages',
'from_poi_to_this_person',
  'from_this_person_to_poi',
    'to_messages'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
my_dataset = data_dict

##df.isnull().sum(axis=0)

##store the data in a data frame
df=pandas.DataFrame.from_dict(my_dataset, orient = 'index')

df= df[features_list]
df = df.replace('NaN', numpy.nan)
## df.info()

## Null values per feature
#df.isnull().sum(axis=0)


## total number of POI
pois = [x for x, y in my_dataset.items() if y['poi']]
## print 'Number of POI\'s: {0}'.format(len(pois))



## total null values
## print "total NaN values: ", df.isnull().sum().sum()

## remove nulls
df.ix[:,:14] = df.ix[:,:14].fillna(0)


from sklearn.preprocessing import Imputer

email_features = [ 'from_messages', 'from_this_person_to_poi', 'from_poi_to_this_person','to_messages']

imp = Imputer(missing_values='NaN', strategy='median', axis=0)

#impute missing values of email features 
df.loc[df[df.poi == 1].index,email_features] = imp.fit_transform(df[email_features][df.poi == 1])
df.loc[df[df.poi == 0].index,email_features] = imp.fit_transform(df[email_features][df.poi == 0])



### Task 2: Remove outliers

#   
# as an example, i plotted the salary vs bonus 
#


for dic in data_dict.values():
    
   matplotlib.pyplot.scatter( dic['salary'] , dic['bonus']  )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


# the plot shows a huge outlier, but lets check out the outler for the salary field

outliers = df.quantile(.975) 
print 'the salaries outlier is :{}'.format(outliers[1])
        
print('\n') 

#for k, v in data_dict.items():
#    if v['salary'] != 'NaN' and v['salary'] > outliers[1]: print 'name : {}'.format(k),':: salary {}'.format(v['salary'])
#print('\n')

outliers = df.quantile(.975) 
print 'the bonus outlier is :{}'.format(outliers[2])
print('\n') 
#for k, v in data_dict.items():
#    if v['bonus'] != 'NaN' and v['bonus'] > outliers[2]: print 'name : {}'.format(k),':: bonus {}'.format(v['bonus'])


# remove the value total  
df = df.drop(['TOTAL','LAVORATO JOHN J'],0)


### Task 3: Create new feature(s)

## new feature that shows the amount of fraction of mails sent to a poi
df['fraction_to_poi'] = df['from_this_person_to_poi']/df['from_messages']

## new feature that shows the amount of fraction of mails sent from a poi
df['fraction_from_poi']=df['from_poi_to_this_person'] / df['to_messages']

#clean all 'infinite' values which ,"division by zero"
df = df.replace('inf', 0)




clf = tree.DecisionTreeClassifier(random_state = 42)
clf.fit(df.ix[:,1:], df.ix[:,:1])

# show the features with non null importance, sorted and create features_list of features for the model
features_importance = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0:
        features_importance.append([df.columns[i+1], clf.feature_importances_[i]])
features_importance.sort(key=lambda x: x[1], reverse = True)
for f_i in features_importance:
    print f_i
features_list = [x[0] for x in features_importance]
features_list.insert(0, 'poi')
#


new_dataset=df.to_dict(orient='index')


new_features_list=['poi','fraction_to_poi', 
'expenses', 
'to_messages',
'total_payments',
'from_messages' ]

data = featureFormat(new_dataset, new_features_list)
labels, features = targetFeatureSplit(data)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
   train_test_split(features, labels, test_size=0.3, random_state=42)

data = featureFormat(new_dataset, new_features_list)
labels, features = targetFeatureSplit(data)
# Provided to give you a starting point. Try a variety of classifiers.

   
#from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report

#############################################################################
###  GaussianNB()                                                           #####
###### Accuracy: 0.84653	Precision: 0.24144	Recall: 0.07050	F1: 0.10913 ######
 #############################################################################

#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
#clf.fit(features_train, labels_train)
#pred=clf.predict(features_test)



 #############################################################################
###  RandomForestClassifier                                                 #####
###### Accuracy: 0.88567	Precision: 0.61373	Recall: 0.38450	F1: 0.47279 #####
 #############################################################################
#clf=RandomForestClassifier()
#clf.fit(features_train, labels_train)
#pred=clf.predict(features_test)

 #############################################################################
###  DecisionTreeClassifier                                                 #####
###### Accuracy: 0.88073	Precision: 0.54969	Recall: 0.58350	F1: 0.56609 #####
 #############################################################################
#clf=tree.DecisionTreeClassifier()
#clf.fit(features_train, labels_train)
#pred=clf.predict(features_test)


#tester.dump_classifier_and_data(clf, new_dataset, features_list)
#tester.main() 

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!




clf = tree.DecisionTreeClassifier(max_features=5,min_samples_split=2,
                                  max_depth = 10,criterion='gini',
                                  random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

tester.dump_classifier_and_data(clf, new_dataset, features_list)
tester.main() 
