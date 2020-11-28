from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt  # data manipulation
import numpy as np
import pandas as pd


from programs import controler
from programs import model_database
pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))
####################################################################################################
#                                   Load project
####################################################################################################
output_path = './output/submission/'
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
y_train = project_analyser.get_train_label()
####################################################################################################
#                                   result functions
####################################################################################################
#Forming a confusion matrix to check our accuracy
def accuracy_calculator(model_name, y_pred, Y_true):
    pp = []
    for p in y_pred:
        if p>0.5:
            pp.append(1)
        else:
            pp.append(0)

    y_pred_f = pp
    cm=confusion_matrix(Y_true,y_pred_f)
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100
    print('{0} model accuracy : {1:.2f} %'.format(model_name, acc))
    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    return y_pred_f, acc

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

model_weight = [0.1, 0.1, 0.15, 0.1, 0.1, 0.15, 0.3]
model_dicty = {'ridgec'         :   model_database.ridgec,
                'lr_elasticnet' :   model_database.lr_elasticnet,
                'svc'           :   model_database.svc,
                'gbc'           :   model_database.gbc,
                'lightgbmc'     :   model_database.lightgbmc,
                'xgboostc'      :   model_database.xgboostc,
                'LogReg'        :   model_database.LogisticRegression,
                'knn'           :   model_database.KNeighborsClassifier,
                'SVC2'          :   model_database.SVC2,
                'decissionTree' :   model_database.DecisionTreeClassifier,
                'adaboost'      :   model_database.AdaBoostClassifier,
                'GradientBoost' :   model_database.GradientBoostingClassifier,
                'GaussianNB'    :   model_database.GaussianNB,
                'RabdomForest'  :   model_database.RandomForestClassifier,
                'ExtraTree'     :   model_database.ExtraTreesClassifier

            }

####################################################################################################
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


def blend_models_predict(X, Y):
    m_predict = []
    for name, m_fit in m_fit_dicty.items():
        predict = m_fit.predict(X)
        accuracy_calculator(name, predict, Y)
        m_predict.append(predict)

    combine = 0
    try:
        for pred, w in zip(m_predict, model_weight):
                combine = combine + (pred * w)
    except:
        for pred in m_predict:
            combine = combine + pred
    return combine

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('\n~~~~~~~~~~~~~~~~~~~~~~For, train data~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
rmse = rmse(y_train, blend_models_predict(X_train, y_train))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('\nrmse score on train data  :~>  ', rmse)

#y_pred = np.floor(np.expm1(blend_models_predict(X_test)))
print('\n\n~~~~~~~~~~~~~~~~~~~~~~For, Test data~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
y_pred = blend_models_predict(X_test, y_test)
y_pred_data = pd.DataFrame({'Outcome':y_pred})
#print(y_pred_data)

######################################## Brutal approach ##########################################
# Brutal approach to deal with predictions close to outer range 
q1 = y_pred_data['Outcome'].quantile(0.0042)
q2 = y_pred_data['Outcome'].quantile(0.99)

y_pred_data['Outcome'] = y_pred_data['Outcome'].apply(lambda x: x if x > q1 else x*0.77)
y_pred_data['Outcome'] = y_pred_data['Outcome'].apply(lambda x: x if x < q2 else x*1.1)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#print('y_pred : ', y_pred[0:10])
print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
y_pred_data['Outcome'], c_acc = accuracy_calculator('Combine', y_pred_data['Outcome'], y_test)
print("\nAccuracy score: %.8f" % (y_test == y_pred_data['Outcome']).mean())
print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#print(pp)
####################################################################################################
#                                   save result
####################################################################################################
y_pred_data.to_csv(output_path+'y_pred_data.csv', index=False)

if actual_split:
        try:
                submission['Outcome'] = y_pred
                path = output+'_with accuracy:_{0}.csv'.format('%.02f' % acc)
                submission.to_csv("./output/submission/"+path, index=False)
                print('result is in output ~>> ', path)
                print('\n\n')
        except:
                pass
else:
        submission['Id']
