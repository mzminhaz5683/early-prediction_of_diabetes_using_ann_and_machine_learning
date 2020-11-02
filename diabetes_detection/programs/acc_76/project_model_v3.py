import pandas as pd
####################################################################################################
#                                   import local
####################################################################################################
# import local files & performance parameters
from programs import controler
try:
    f_counter = open('./output/model_counter.txt', 'r')
    counter = int(f_counter.read())
    f_counter.close()
except:
    counter = 0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
f_counter = open('./output/model_counter.txt', 'w')
if controler.model_counter_reset:
    counter = 0
else:
    counter = str(counter+1)
f_counter.write(counter)
f_counter.close()
####################################################################################################
#                                   import documents
####################################################################################################
submission = pd.read_csv("./input/sample_submission.csv")
from programs import project_v2 as project_analyser

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
output = 'model'
file_formate = '_apply_'+counter+'.csv'
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

#Applying logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#Calculating Y_predection
y_pred=classifier.predict(X_test)

####################################################################################################
#                                   finding accuracy
####################################################################################################
#Forming a confusion matrix to check our accuracy
from sklearn.metrics import confusion_matrix
actual_result = pd.read_csv("./input/actual_result.csv")
y_test = actual_result['Outcome']

cm=confusion_matrix(y_test,y_pred)
print('\n ______________________________________________________________________\n')
print('\n ____________________ model ends with the result ______________________\n')
print(cm)
acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100
print("\n   :::: accuracy :::: \n\n        ({0}+{3})\n _________________________\n\n    ({0}+{1}+{2}+{3})) * 100 = {4} %"
        .format(cm[0][0], cm[0][1], cm[1][0], cm[1][1], '%.02f' % acc))
####################################################################################################
#                                   save result
####################################################################################################
submission['Outcome'] = y_pred

submission.to_csv("./output/submission/"+output+file_formate, index=False)
print('\n ______________________________________________________________________ \n')
print('Model ends for count: {0} sucessfully & saves result in output with the name'.format(counter))
print(' ~>> '+output+file_formate)
print('\n\n')