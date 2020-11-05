from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt  # data manipulation
import numpy as np
import pandas as pd


from programs import controler
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
#                                   accuracy function
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
####################################################################################################
#                                   Model Start : 1
####################################################################################################
from sklearn.ensemble import VotingClassifier
from mlens.ensemble import SuperLearner
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from mlxtend.classifier import StackingCVClassifier
if 0:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_models(lr, knn, svm, dt, ab, gb, g_nb, rf, et):
        """Generate a library of base learners."""
        models = {}

        if lr:
            param1 = {'C': 0.7678243129497218, 'penalty': 'l1', 'solver': 'liblinear', 'random_state': 10}
            model1 = LogisticRegression(**param1)
            models['Logistic Regression'] = model1
        
        if knn:
            param2 = {'n_neighbors': 4, 'metric': 'minkowski', 'p':2}
            model2 = KNeighborsClassifier(**param2)
            models['K Neighbors'] = model2

        if svm:
        # kernel: linear
            param3 = {'C': 1.7, 'kernel': 'rbf', 'random_state':  10, 'probability':True}
            model3 = SVC(**param3)
            models['Support Vector'] = model3

        if dt:
            #param = {'criterion': 'gini', 'max_depth': 3, 'max_features': 2, 'min_samples_leaf': 3}
            param4 = {'criterion': 'gini', 'max_depth': 3, 'random_state': 10}
            model4 = DecisionTreeClassifier(**param4)
            models['Decision Tree'] = model4

        if ab:
            param5 = {'learning_rate': 0.05, 'n_estimators': 150}
            model5 = AdaBoostClassifier(**param5)
            models['Ada Boost'] = model5

        if gb:
            param6 = {'learning_rate': 0.01, 'n_estimators': 100}
            model6 = GradientBoostingClassifier(**param6)
            models['Gradient Boost'] = model6

        if g_nb:
            model7 = GaussianNB()
            models['Naive Bayes (Gaussian)'] = model7

        if rf:
            param8 = {'n_estimators': 15, 'criterion': 'gini', 'random_state':10}
            model8 = RandomForestClassifier(**param8)
            models['Random Forest'] = model8

        if et:
            model9 = ExtraTreesClassifier()
            models['Extra Trees'] = model9
                
        return models
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Forming a confusion matrix to check our accuracy
    def accuracy_calculator(y_pred):
            cm=confusion_matrix(y_test,y_pred)
            acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100
            return acc
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def prediction(model_list,X_train, X_test, y_train, y_test):
        prediction_set = []
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        for i, (name, model) in enumerate(models.items()):
            print('Predict for ~> ', name)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            #print('predict : ', y_pred[0:10])

            #y_pred2 = model.predict_proba(X_test)
            #print('predict_proba : ', y_pred2[0:10])

            m_acc = accuracy_calculator(y_pred)
            prediction_set.append(y_pred)

            print('Model accuracy : {0:.2f} %'.format(m_acc))
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

        return prediction_set
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # lr, knn, svm, dt, ab, gb, g_nb, rf, et :: 9
    models = get_models(1, 0, 0, 1, 1, 0, 1, 1, 0)

    if 0:
        prediction_set = prediction(models,X_train,X_test,y_train,y_test)
    else:
        base_learners = models
        meta_learner = GradientBoostingClassifier(
            n_estimators=100, # 1000
            loss="exponential",
            # max_features=6,
            max_depth=3,
            subsample=0.5,
            learning_rate=0.001, 
            random_state=10
        )

        # Instantiate the ensemble with 10 folds
        sl = SuperLearner(
            folds=10,
            random_state=10,
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
                pp.append(1)
            else:
                pp.append(0)

        print("\nSuper Learner Accuracy score: %.8f" % (y_test== pp).mean())
        print(pp)
        c_acc = accuracy_calculator(pp)
        print('Combine model accuracy : {0:.2f} %'.format(c_acc))
else:
    ####################################################################################################
    #                                   model start : 2
    ####################################################################################################
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline

    
    kfolds = KFold(n_splits=10, shuffle=True, random_state=controler.rndm_state)
    # rmse
    def rmse(y_train, y_pred):
        return np.sqrt(mean_squared_error(y_train, y_pred))
    
    # build our model scoring function
    def cv_rmse(model):
        rmse = np.sqrt(-cross_val_score(model, X_train, y_train,
                                            scoring="neg_mean_squared_error",
                                            cv=kfolds))
        return rmse

    # setup models
    alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
    alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
    e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
    e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]


    ridgec = make_pipeline(RobustScaler(), RidgeClassifierCV(alphas=alphas_alt,
                                                    cv=kfolds))


    lr_elasticnet = make_pipeline(RobustScaler(), LogisticRegression(penalty = 'elasticnet',
                                                    solver = 'saga',
                                                    max_iter=1e7,
                                                    l1_ratio=0.04,
                                                    random_state=controler.rndm_state)) #alphas=e_alphas


    svc = make_pipeline(RobustScaler(), SVC(C=20,
                                            gamma=0.0003,
                                            random_state=controler.rndm_state))


    gbc = GradientBoostingClassifier(learning_rate=0.05,
                                    max_depth=4,
                                    max_features='sqrt',
                                    min_samples_leaf=15,
                                    min_samples_split=10,
                                    loss='exponential', # Adaboost
                                    random_state=controler.rndm_state,
                                    n_estimators=controler.n_estimators)


    lightgbmc = LGBMClassifier(objective='binary',
                                num_leaves=4,
                                learning_rate=0.01,
                                max_bin=200,
                                bagging_fraction=0.75,
                                bagging_freq=5,
                                bagging_seed=7,
                                feature_fraction=0.2,
                                feature_fraction_seed=7,
                                verbose=-1,
                                n_estimators=controler.n_estimators + 2000)

    
    xgboostc = XGBClassifier(learning_rate=0.01,
                            max_depth=3,
                            min_child_weight=0,
                            gamma=0,
                            subsample=0.7,
                            colsample_bytree=0.7,
                            nthread=-1,
                            scale_pos_weight=1,
                            seed=27,
                            reg_alpha=0.00006,
                            random_state=controler.rndm_state,
                            n_estimators=controler.n_estimators + 460)
    

    # ridgec, lr_elasticnet, svc, gbc, lightgbmc, xgboostc
    stack_gen = StackingCVClassifier(classifiers=(ridgec, lr_elasticnet, svc, gbc, lightgbmc, xgboostc),
                                    meta_classifier=xgboostc,
                                    use_features_in_secondary=True)
    
    #~~~~~~~~~~~~~~~~~~~ Modeling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('\n\n~~~~~~~~~~~~~~~~~TEST score on Cross Validation~~~~~~~~~~~~~~~~~')

    score = cv_rmse(ridgec)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print("Kernel Ridge Classifier score: mean:{:.4f},  std:{:.4f}\n".format(score.mean(), score.std()) )

    score = cv_rmse(lr_elasticnet)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print("Logistic Regression: ElasticNet score: mean:{:.4f},  std:{:.4f}\n".format(score.mean(), score.std()) )

    score = cv_rmse(svc)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print("Support Vector Classifier score: mean:{:.4f},  std:{:.4f}\n".format(score.mean(), score.std()) )

    score = cv_rmse(gbc)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print("Gradient Boosting Classifier score: mean:{:.4f},  std:{:.4f}\n".format(score.mean(), score.std()) )

    score = cv_rmse(lightgbmc)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print("lightgbm Classifier score: mean:{:.4f},  std:{:.4f}\n".format(score.mean(), score.std()) )

    score = cv_rmse(xgboostc)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print("XGBoost Classifier score: mean:{:.4f},  std:{:.4f}\n".format(score.mean(), score.std()) )

    
    # ridgec, lr_elasticnet, svc, gbc, lightgbmc, xgboostc
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Start : StackingCV Classifier')
    stack_gen_model = stack_gen.fit(np.array(X_train), np.array(y_train))
    
    print('Start : ridgec')
    ridgec_model_full_data = ridgec.fit(X_train, y_train)
    
    print('Start : lr_elasticnet')
    lr_elasticnet_model_full_data = lr_elasticnet.fit(X_train, y_train)
    
    print('Start : svc')
    svc_model_full_data = svc.fit(X_train, y_train)
    
    print('Start : GradientBoostingClassifier')
    gbc_model_full_data = gbc.fit(X_train, y_train)
    
    print('Start : lightgbmc')
    lightgbmc_model_full_data = lightgbmc.fit(X_train, y_train)
    
    print('Start : xgboostc')
    xgboostc_model_full_data = xgboostc.fit(X_train, y_train)

    # ridgec, lr_elasticnet, svc, gbc, lightgbmc, xgboostc
    def blend_models_predict(X, Y):
        rd_p    = ridgec_model_full_data.predict(X)
        accuracy_calculator('ridgec', rd_p, Y)
        lr_p    = lr_elasticnet_model_full_data.predict(X)
        accuracy_calculator('lr_elasticnet', lr_p, Y)
        svc_p   = svc_model_full_data.predict(X)
        accuracy_calculator('svc', svc_p, Y)
        gdc_p   = gbc_model_full_data.predict(X)
        accuracy_calculator('gbc', gdc_p, Y)
        lgbmc_p = lightgbmc_model_full_data.predict(X)
        accuracy_calculator('lightgbmc', lgbmc_p, Y)
        xgbc_p  = xgboostc_model_full_data.predict(X)
        accuracy_calculator('xgboostc', xgbc_p, Y)
        sg_p    = stack_gen_model.predict(np.array(X))
        accuracy_calculator('stack_gen', sg_p, Y)

        w = [0.1, 0.1, 0.15, 0.1, 0.1, 0.15, 0.3]
        return ((rd_p*w[0])+(lr_p*w[1])+(svc_p*w[2])+(gdc_p*w[3])+(lgbmc_p*w[4])+(xgbc_p*w[5])
                +(sg_p*w[6]))

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
