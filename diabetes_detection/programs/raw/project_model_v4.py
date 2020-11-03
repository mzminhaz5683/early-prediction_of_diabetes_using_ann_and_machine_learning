from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt # data manipulation
import pandas as pd
import numpy as np

from programs import controler

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
output = 'model'
from programs import project_v4 as project_analyser
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


####################################################################################################
#                                   Load documents
####################################################################################################
print('______________________________________________________________________\n')
print('_____________________________ model start ____________________________\n')
print('______________________________________________________________________\n')
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
lst = []
dtc_acc, dtc_y  = accuracy_calculator(decisionTreeClassifier())
print('dtc_acc : {0:.2f} %'.format(dtc_acc))
lst.append(dtc_acc)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
lr_acc, lr_y    = accuracy_calculator(logisticRegression())
print('lr_acc : {0:.2f} %'.format(lr_acc))
lst.append(lr_acc)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
rf_acc, rf_y    = accuracy_calculator(randomForest())
print('rf_acc : {0:.2f} %'.format(rf_acc))
lst.append(rf_acc)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
knn_acc, knn_y  = accuracy_calculator(kNeighbors())
print('knn_acc : {0:.2f} %'.format(knn_acc))
lst.append(knn_acc)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nb_acc, nb_y    = accuracy_calculator(naiveBayes())
print('nb_acc : {0:.2f} %'.format(nb_acc))
lst.append(nb_acc)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
svm_acc, svm_y  = accuracy_calculator(svm())
print('svm_acc : {0:.2f} %'.format(svm_acc))
lst.append(svm_acc)


y_pred = svm_y
####################################################################################################
#                                   save result
####################################################################################################
if actual_split:
        submission['Outcome'] = y_pred
        path = output+'_with accuracy:_{0}.csv'.format('%.02f' % acc)
        submission.to_csv("./output/submission/"+path, index=False)
        print('______________________________________________________________________ \n')
        print('Model ends & saves result in output with the name')
        print(' ~>> ', path)
        print('\n\n')