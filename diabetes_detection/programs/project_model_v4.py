from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt # data manipulation
import pandas as pd
import numpy as np

from programs import controler
####################################################################################################
#                                   Load project
####################################################################################################
output = 'model'
project = ''
if controler.project_version == 3:
        from programs import project_v3_actual_split as project_analyser
        project = 'project_v3_actual_split'
elif controler.project_version == 4:
        from programs import project_v4_random_split as project_analyser
        project = 'project_v4_random_split'
####################################################################################################
#                                   Load documents
####################################################################################################
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('             model start for : {0}'.format(project))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
submission = pd.read_csv("./input/sample_submission.csv")
actual_split = 1
try:
        y_test = project_analyser.get_actual_result()
        actual_split = 0
except:
        actual_result = pd.read_csv("./input/actual_result.csv")
        y_test = actual_result['Outcome'] # actual result

X_train, X_test = project_analyser.get_train_test_data()
y = y_train = project_analyser.get_train_label()
####################################################################################################
#                                   Model Start
####################################################################################################
#Applying Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Appling random forest
def randomForest():
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=controler.n_estimators,\
                criterion=controler.criterion,random_state=controler.rndm_state) # 10, entropy, 0
        classifier.fit(X_train,y_train)

        #Calculating Y_predection
        return classifier.predict(X_test)



#Appling K - Neighbors
def kNeighbors():
        from sklearn.neighbors import KNeighborsClassifier as KNC
        classifier = KNC(n_neighbors=4, metric='minkowski', p=2) # p=1
        classifier.fit(X_train,y_train)

        #Calculating Y_predection
        return classifier.predict(X_test)



# Appling SVM
def svm():
        from sklearn.svm import SVC
        classifier = SVC(kernel ='rbf',random_state=controler.rndm_state) # linear, 0
        classifier.fit(X_train,y_train)

        #Calculating Y_predection
        return classifier.predict(X_test)



#Applying Naive Bayes Classifier
def naiveBayes():
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train,y_train)

        #Calculating Y_predection
        return classifier.predict(X_test)



#Applying logistic Regression
def logisticRegression():
        from sklearn.linear_model import LogisticRegression
        classifier=LogisticRegression(random_state=controler.rndm_state) # 0, 
        classifier.fit(X_train,y_train)

        #Calculating Y_predection
        return classifier.predict(X_test)



# Applying Decision Tree classifer
def decisionTreeClassifier():
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion=controler.criterion, max_depth=3,\
                random_state=controler.rndm_state) # gini, 3, 0
        classifier.fit(X_train,y_train)

        #Calculating Y_predection
        return classifier.predict(X_test)


#Forming a confusion matrix to check our accuracy
def accuracy_calculator(y_pred):
        cm=confusion_matrix(y_test,y_pred)
        print('______________________________\n')
        acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100
        return acc, y_pred

####################################################################################################
#                                   finding accuracy
####################################################################################################
dectionary = {}
dectionary_y = {}
dtc_acc, dtc_y  = accuracy_calculator(decisionTreeClassifier())
print('dtc_acc : {0:.2f} %'.format(dtc_acc))
dectionary['dtc_acc'] = dtc_acc
dectionary_y['dtc_acc'] = dtc_y
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
lr_acc, lr_y    = accuracy_calculator(logisticRegression())
print('lr_acc : {0:.2f} %'.format(lr_acc))
dectionary['lr_acc'] = lr_acc
dectionary_y['lr_acc'] = lr_y
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
rf_acc, rf_y    = accuracy_calculator(randomForest())
print('rf_acc : {0:.2f} %'.format(rf_acc))
dectionary['rf_acc'] = rf_acc
dectionary_y['rf_acc'] = rf_y
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
knn_acc, knn_y  = accuracy_calculator(kNeighbors())
print('knn_acc : {0:.2f} %'.format(knn_acc))
dectionary['knn_acc'] = knn_acc
dectionary_y['knn_acc'] = knn_y
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nb_acc, nb_y    = accuracy_calculator(naiveBayes())
print('nb_acc : {0:.2f} %'.format(nb_acc))
dectionary['nb_acc'] = nb_acc
dectionary_y['nb_acc'] = nb_y
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
svm_acc, svm_y  = accuracy_calculator(svm())
print('svm_acc : {0:.2f} %'.format(svm_acc))
dectionary['svm_acc'] = svm_acc
dectionary_y['svm_acc'] = svm_y
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('\n\n~~~~~~~~~~~~~~~~~~~~~~~ final accuracy ~~~~~~~~~~~~~~~~~~~~~~~')

# selection sort : accending
lst = [i for i in dectionary.keys()]
for i in range(0, len(lst)):
        min = i
        for j in range(i+1, len(lst)):
                if dectionary[lst[min]] > dectionary[lst[j]]:
                        min = j
        temp = lst[min]
        lst[min]=lst[i]
        lst[i] = temp
print('lst = ', lst)
if controler.combine_machines:
        result_y =      np.ceil(dectionary_y[lst[5]]   *       0.40
                                +dectionary_y[lst[4]]   *       0.20
                                +dectionary_y[lst[3]]   *       0.15
                                +dectionary_y[lst[2]]   *       0.10
                                +dectionary_y[lst[1]]   *       0.09
                                +dectionary_y[lst[0]]   *       0.06
                        )
        acc, y_pred  = accuracy_calculator(result_y)
        print('combine : {0:.2f} %'.format(acc))
else:
        acc = dectionary[lst[-1]]
        y_pred = dectionary_y[lst[-1]]
        print('{0} : {1:.2f} %'.format(lst[-1], acc))
print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('             model ends for : {0}'.format(project))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
####################################################################################################
#                                   save result
####################################################################################################
if actual_split:
        submission['Outcome'] = y_pred
        path = output+'_with accuracy:_{0}.csv'.format('%.02f' % acc)
        submission.to_csv("./output/submission/"+path, index=False)
        print('result is in output ~>> ', path)
        print('\n\n')
else:
        submission['Id']