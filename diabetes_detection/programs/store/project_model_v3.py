from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt # data manipulation
import pandas as pd
import numpy as np


from programs import project_v3 as project_analyser
from programs import controler
####################################################################################################
#                                   Load documents
####################################################################################################
print('______________________________________________________________________\n')
print('_____________________________ model start ____________________________\n')
print('______________________________________________________________________\n')
submission = pd.read_csv("./input/sample_submission.csv")
actual_result = pd.read_csv("./input/actual_result.csv")
y_test = actual_result['Outcome'] # actual result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
output = 'model'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
        classifier.fit(X_train,y_train)

        #Calculating Y_predection
        return classifier.predict(X_test)



#Appling K - Neighbors
def kNeighbors():
        from sklearn.neighbors import KNeighborsClassifier as KNC
        classifier = KNC(n_neighbors=4, metric='minkowski', p=1)
        classifier.fit(X_train,y_train)

        #Calculating Y_predection
        return classifier.predict(X_test)



# Appling SVM
def svm():
        from sklearn.svm import SVC
        classifier = SVC(kernel ='linear',random_state=0)
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
        classifier=LogisticRegression(random_state=0)
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
lr_acc, lr_y    = accuracy_calculator(logisticRegression())
print('lr_acc : {0:.2f} %'.format(lr_acc))
rf_acc, rf_y    = accuracy_calculator(randomForest())
print('rf_acc : {0:.2f} %'.format(rf_acc))
knn_acc, knn_y  = accuracy_calculator(kNeighbors())
print('knn_acc : {0:.2f} %'.format(knn_acc))
nb_acc, nb_y    = accuracy_calculator(naiveBayes())
print('nb_acc : {0:.2f} %'.format(nb_acc))
svm_acc, svm_y  = accuracy_calculator(svm())
print('svm_acc : {0:.2f} %'.format(svm_acc))


if 0:
        lr      = int(lr_acc)
        rf      = int(rf_acc)
        knn     = int(knn_acc)
        nb      = int(nb_acc)
        svm     = int(svm_acc)
        #Comparring results from different Algorithms
        objects = ('Random Forest', 'K-NN', 'SVM', 'Naive Bayes', 'Logistic Regression')
        y_pos = np.arange(len(objects))
        performance = [rf,knn,svm,nb,lr]
        
        plt.scatter(y_pos, performance, alpha=1)
        plt.plot(y_pos, performance,color='blue')
        plt.xticks(y_pos, objects)
        plt.ylabel('Accuracy %')
        plt.xticks(rotation=45)
        plt.title('Algorithm Accuracy')
        plt.show()


acc = svm_acc
y_pred = svm_y
####################################################################################################
#                                   save result
####################################################################################################
submission['Outcome'] = y_pred
path = output+'_with accuracy:_{0}.csv'.format('%.02f' % acc)
submission.to_csv("./output/submission/"+path, index=False)
print('______________________________________________________________________ \n')
print('Model ends & saves result in output with the name')
print(' ~>> ', path)
print('\n\n')