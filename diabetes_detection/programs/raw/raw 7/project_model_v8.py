from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt  # data manipulation
import numpy as np
import pandas as pd


from programs import controler

####################################################################################################
#                                   Load project
####################################################################################################
output = 'model'
SEED = 7
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
from sklearn.ensemble import VotingClassifier
from mlens.ensemble import SuperLearner
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_models():
    """Generate a library of base learners."""

    param1 = {'C': 0.7678243129497218, 'penalty': 'l1', 'solver': 'liblinear', 'random_state': 10}
    model1 = LogisticRegression(**param1)

    param2 = {'n_neighbors': 4, 'metric': 'minkowski', 'p':2}
    model2 = KNeighborsClassifier(**param2)

    # kernel: linear
    param3 = {'C': 1.7, 'kernel': 'rbf', 'random_state':  10, 'probability':True}
    model3 = SVC(**param3)

    #param = {'criterion': 'gini', 'max_depth': 3, 'max_features': 2, 'min_samples_leaf': 3}
    param4 = {'criterion': 'gini', 'max_depth': 3, 'random_state': 10}
    model4 = DecisionTreeClassifier(**param4)

    param5 = {'learning_rate': 0.05, 'n_estimators': 150}
    model5 = AdaBoostClassifier(**param5)

    param6 = {'learning_rate': 0.01, 'n_estimators': 100}
    model6 = GradientBoostingClassifier(**param6)

    model7 = GaussianNB()

    param8 = {'n_estimators': 15, 'criterion': 'gini', 'random_state':10}
    model8 = RandomForestClassifier(**param8)

    model9 = ExtraTreesClassifier()

    models = {'Logistic Regression':model1, 'K Neighbors':model2, 'Support Vector':model3,
              'Decision Tree':model4, 'Ada Boost':model5, 'Gradient Boost':model6,
              'Naive Bayes (Gaussian)':model7, 'Random Forest':model8,  'Extra Trees':model9}
              
    return models
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Forming a confusion matrix to check our accuracy
def accuracy_calculator(y_pred):
        cm=confusion_matrix(y_test,y_pred)
        acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100
        return acc
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def prediction(model_list,X_train, X_test, y_train, y_test):
    if 1:
        prediction_set = []
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        for i, (name, model) in enumerate(models.items()):
            print('Predict for ~> ', name)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            print('predict : ', y_pred[0:10])

            y_pred2 = model.predict_proba(X_test)
            print('predict_proba : ', y_pred2[0:10])

            #y_pred1 = model.predict_log_proba(X_test)
            #print('predict_log_proba : ', y_pred1[0:10])

            m_acc = accuracy_calculator(y_pred)
            prediction_set.append(y_pred)

            print('Model accuracy : {0:.2f} %'.format(m_acc))
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

    else:
        """Fit models in list on training set and return preds"""
        P = np.zeros((y_test.shape[0], len(model_list)))
        P = pd.DataFrame(P)

        print("Fitting models.")
        cols = list()
        for i, (name, m) in enumerate(models.items()):
            print("%s..." % name, end=" ", flush=False)
            m.fit(X_train, y_train)
            P.iloc[:, i] = m.predict_proba(X_test)[:, 1]
            cols.append(name)
            print("done")

        P.columns = cols
        print("Done.\n")
        return P
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if 1:
    models = get_models()
    P = prediction(models,X_train,X_test,y_train,y_test)
else:
    base_learners = get_models()
    meta_learner = GradientBoostingClassifier(
        n_estimators=1000,
        loss="exponential",
        max_features=6,
        max_depth=3,
        subsample=0.5,
        learning_rate=0.001, 
        random_state=SEED
    )

    # Instantiate the ensemble with 10 folds
    sl = SuperLearner(
        folds=10,
        random_state=SEED,
        verbose=2,
        backend="multiprocessing"
    )

    # Add the base learners and the meta learner
    sl.add(list(base_learners.values()), proba=True) 
    sl.add_meta(meta_learner, proba=True)

    # Train the ensemble
    sl.fit(X_train, y_train)

    # Predict the test set
    p_sl = sl.predict_proba(X_test)

    # print("\nSuper Learner ROC-AUC score: %.3f" % roc_auc_score(y_test_sc, p_sl[:, 1]))

    pp = []
    for p in p_sl[:, 1]:
        if p>0.5:
            pp.append(1.)
        else:
            pp.append(0.)

    print("\nSuper Learner Accuracy score: %.8f" % (y_test== pp).mean())
    print(p_sl)
    #accuracy_calculator(p_sl)

####################################################################################################
#                                   save result
####################################################################################################
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
