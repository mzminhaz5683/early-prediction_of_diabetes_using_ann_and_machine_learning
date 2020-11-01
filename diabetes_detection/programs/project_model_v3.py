import pandas as pd
####################################################################################################
#                                   import local
####################################################################################################
# import local files & performance parameters
# noinspection PyBroadException
try:
    f_counter = open('./output/contents/model_counter.txt', 'r')
    counter = int(f_counter.read())
    f_counter.close()
except:
    counter = 0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
f_counter = open('./output/contents/model_counter.txt', 'w')
counter = str(counter+1)
f_counter.write(counter)
f_counter.close()
####################################################################################################
#                                   import documents
####################################################################################################
submission = pd.read_csv("./input/sample_submission.csv")
from programs import project_v2 as project_analyser
output = 'project_v2'
file_formate = '_m2_'+counter+'.csv'
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Applying logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
y_pred=classifier.predict(X_test)

#Forming a confusion matrix to check our accuracy
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
















####################################################################################################
#                                   save result
####################################################################################################
submission.to_csv("./output/"+output+file_formate, index=False)
print('\n\n\n________________Stage finished : {0}___________________'.format(counter))
print('\n\nSubmissin sucessfull saved in output with the name')
print(' ~>> '+output+file_formate)
print('\n\n__________________________________________________________')