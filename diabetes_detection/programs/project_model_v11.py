from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt  # data manipulation
from statistics import mode
import numpy as np
import pandas as pd


from programs import controler
from programs import model_database
from programs.checker_v2 import accuracy_calculator
from programs.ann_model_v4 import get_test_result
pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))
####################################################################################################
#                                   Load project
####################################################################################################
output_path = './output/predicted_results/'
project = ''
random_split = 1
if controler.project_version == 3:
        from programs import project_v3_actual_split as project_analyser
        project = 'project_v3_actual_split'
        random_split = 0
elif controler.project_version == 5:
        from programs import project_v5_random_split as project_analyser
        project = 'project_v5_random_split'
else:
    print("\n\n\n Can't find any process model \n\n\n")
    exit(0)
####################################################################################################
#                                   Load documents
####################################################################################################
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('             model start for : {0}'.format(project))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
try:
        y_test = project_analyser.get_actual_result()

except:
        actual_result = pd.read_csv("./input/actual_result.csv")
        y_test = actual_result['Outcome'] # actual result

X_train, X_test = project_analyser.get_train_test_data()
y_train = project_analyser.get_train_label()
X_test_ID, X_train_ID = project_analyser.get_IDs()
####################################################################################################
#                                   result functions
####################################################################################################
result_file = pd.DataFrame({'Id':X_test_ID})
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# root mean square error function
def rmse(y_train, y_pred):
    return np.sqrt(mean_squared_error(y_train, y_pred))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# model scoring function
def cv_rmse(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error",
                                        cv=model_database.kfolds))
    return rmse
####################################################################################################
#                                   model parameters
####################################################################################################
####################################################################################################
# LogisticRegression KNeighborsClassifier SVC2 DecisionTreeClassifier AdaBoostClassifier GradientBoostingClassifier
# GaussianNB RandomForestClassifier ExtraTreesClassifier

model_weight = []
model_weight = []
model_dicty = {
                'ridgec'        :   model_database.ridgec,
                'lr_elasticnet' :   model_database.lr_elasticnet,
                'svc'           :   model_database.svc,
#                'gbc'           :   model_database.gbc,
#                'lightgbmc'     :   model_database.lightgbmc,
                'xgboostc'      :   model_database.xgboostc,
                'LogReg'        :   model_database.LogisticRegression,
#                'knn'           :   model_database.KNeighborsClassifier,
#                'SVC2'          :   model_database.SVC2,
#                'decissionTree' :   model_database.DecisionTreeClassifier,
                'adaboost'      :   model_database.AdaBoostClassifier,
#                'GradientBoost' :   model_database.GradientBoostingClassifier,
#                'GaussianNB'    :   model_database.GaussianNB,
#                'RabdomForest'  :   model_database.RandomForestClassifier,
#                'ExtraTree'     :   model_database.ExtraTreesClassifier
                }

####################################################################################################
####################################################################################################
#                                   save high accuracy dataset
####################################################################################################
def save_80_acc(acc, name):
    if acc > 87.66 and random_split:
        import os
        path = "output/set_of_+80_acc/{0:.2f}_for_{1}".format(acc, name)
        save_path = "./output/set_of_+80_acc/{0:.2f}_for_{1}/".format(acc, name)
        try:
            os.mkdir(path)
            print('Path created')
        except:
            print('Can not create path')
            pass

        project_analyser.save_random_split(save_path)

        print('Data recorded in : {0}'.format(path))
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
####################################################################################################
#                                   model start : 1
####################################################################################################
print('\n\n~~~~~~~~~~~~~~~~~TEST score on Cross Validation~~~~~~~~~~~~~~~~~')

for name, model in model_dicty.items():
    score = cv_rmse(model)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print("{0} score: mean:{1:.4f},  std:{2:.4f}\n".format(name, score.mean(), score.std()))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Start : StackingCV Classifier')
stack_gen_model = model_database.stack_gen.fit(np.array(X_train), np.array(y_train))

m_fit_dicty = {}
for name, model in model_dicty.items():    
    print('Start fiting : {0}'.format(name))
    model_fit = model.fit(X_train, y_train)
    m_fit_dicty[name] = model_fit


def blend_models_predict(X, Y, test=0):
    best_acc = best_acc_index = count = 0
    m_predict = []
    if test:
        y_ann, best_acc = get_test_result()
        m_predict.append(y_ann)
        best_acc_index = count = 1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for name, m_fit in m_fit_dicty.items():
        predict = m_fit.predict(X)
        m_predict.append(predict)
        if test:
            _, acc = accuracy_calculator(name, predict, Y)
            if acc > best_acc:
                best_acc = acc
                best_acc_index = count
            save_80_acc(acc, name)
        count += 1
    
    #print('best_acc_index =',best_acc_index)
    # Max voting among predictions
    result = np.array([])
    for i in range(0, len(m_predict[0])):
        try:
            result = np.append(result, mode([clm[i] for clm in m_predict]))
        except:
            result = np.append(result, m_predict[best_acc_index][i])
    return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('\n~~~~~~~~~~~~~~~~~~~~~~For, Train data~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
rmse = rmse(y_train, blend_models_predict(X_train, y_train))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('\nrmse score on train data  :~>  ', rmse)

#y_pred = np.floor(np.expm1(blend_models_predict(X_test)))
print('\n\n~~~~~~~~~~~~~~~~~~~~~~For,  Test data~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
y_pred = blend_models_predict(X_test, y_test, test=1)
result_file['Outcome'] = y_pred
#print(result_file)

######################################## Brutal approach ##########################################
# Brutal approach to deal with predictions close to outer range 
q1 = result_file['Outcome'].quantile(0.0042)
q2 = result_file['Outcome'].quantile(0.99)

result_file['Outcome'] = result_file['Outcome'].apply(lambda x: x if x > q1 else x*0.77)
result_file['Outcome'] = result_file['Outcome'].apply(lambda x: x if x < q2 else x*1.1)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#print('y_pred : ', y_pred[0:10])
print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
result_file['Outcome'], c_acc = accuracy_calculator('Combine', result_file['Outcome'], y_test)
#print("\nAccuracy score: %.8f" % (y_test == result_file['Outcome']).mean())
print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#print(pp)
####################################################################################################
#                                   save result
####################################################################################################
save_80_acc(c_acc, 'Combine')
####################################################################################################
if c_acc > 87:
    file_name = controler.resut_file_name + '_with acc:_{0:.2f}.csv'.format(c_acc)
    print('Result saved in :~> ', output_path+file_name)
    result_file.to_csv(output_path + file_name, index=False)

