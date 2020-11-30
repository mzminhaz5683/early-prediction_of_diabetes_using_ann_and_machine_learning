from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Classifiers~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.ensemble import VotingClassifier
from mlens.ensemble import SuperLearner
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from mlxtend.classifier import StackingCVClassifier
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt  # data manipulation
import numpy as np
import pandas as pd
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from programs import controler

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
kfolds = KFold(n_splits=10, shuffle=True, random_state=controler.rndm_state)

# setup models
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]


ridgec = make_pipeline(RobustScaler(), RidgeClassifierCV(alphas=alphas_alt, cv=kfolds))


lr_elasticnet = make_pipeline(RobustScaler(), LogisticRegression(penalty = 'elasticnet',
                                                solver = 'saga',
                                                max_iter=1e7,
                                                l1_ratio=0.05, # 0.04
                                                random_state=controler.rndm_state))


svc = make_pipeline(RobustScaler(), SVC(C=20,
                                        gamma=0.0003, # 0.0003
                                        random_state=controler.rndm_state))


gbc = GradientBoostingClassifier(learning_rate=0.01, #0.05
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=13, # 15
                                min_samples_split=10, # 10
                                loss='exponential', # Adaboost, exponential
                                random_state=controler.rndm_state,
                                n_estimators=controler.n_estimators)


lightgbmc = LGBMClassifier(objective='binary',
                            num_leaves=4,
                            learning_rate=0.01, # 0.01
                            max_bin=200,
                            bagging_fraction=0.75, # 0.75
                            bagging_freq=5, # 5
                            bagging_seed=7, # 7
                            feature_fraction=0.2,
                            feature_fraction_seed=7,
                            verbose=-1,
                            n_estimators=controler.n_estimators + 2000)


xgboostc = XGBClassifier(learning_rate=0.01, # 0.01
                        max_depth=3,
                        min_child_weight=0,
                        gamma=0.0000, # 0
                        subsample=0.7,
                        colsample_bytree=0.7,
                        nthread=-1,
                        scale_pos_weight=1,
                        seed=27, # 27
                        reg_alpha= 0.0005, # 0.00006,
                        random_state=controler.rndm_state,
                        n_estimators=controler.n_estimators + 460)


###################################################################################################
param1 = {'C': 0.7678, 'penalty': 'l1', 'solver': 'liblinear', 'random_state': controler.rndm_state}
lr = LogisticRegression = LogisticRegression(**param1)

param2 = {'n_neighbors': 4, 'metric': 'minkowski', 'p':2}
knn = KNeighborsClassifier = KNeighborsClassifier(**param2)

param3 = {'C': 1.7, 'kernel': 'rbf', 'random_state':  controler.rndm_state, 'probability':True}
SVC2 = SVC(**param3)

#param = {'criterion': 'gini', 'max_depth': 3, 'max_features': 2, 'min_samples_leaf': 3}
param4 = {'criterion': 'gini', 'max_depth': 3, 'max_features': 2, 'min_samples_leaf': 3,
                                                'random_state': controler.rndm_state}
dt = DecisionTreeClassifier = DecisionTreeClassifier(**param4)

#param5 = {'learning_rate': 0.05, 'n_estimators': 150}
param5 = {'learning_rate': 0.04, 'n_estimators': 150}
adab = AdaBoostClassifier = AdaBoostClassifier(**param5)

param6 = {'learning_rate': 0.01, 'n_estimators': 100}
gbc2 = GradientBoostingClassifier = GradientBoostingClassifier(**param6)

gnb = GaussianNB = GaussianNB()

param8 = {'n_estimators': 15, 'criterion': 'gini', 'random_state':controler.rndm_state}
rf = RandomForestClassifier = RandomForestClassifier(**param8)

et = ExtraTreesClassifier = ExtraTreesClassifier()

####################################################################################################

# ridgec, lr_elasticnet, svc, gbc, lightgbmc, xgboostc
# lr, knn, SVC2, dt, adab, gbc2, gnb, rf, et
stack_gen = StackingCVClassifier(classifiers=(ridgec, lr_elasticnet, svc, gbc, xgboostc, dt, adab, rf, et),
                                meta_classifier=xgboostc,
                                use_features_in_secondary=True)


'''# ridgec, lr_elasticnet, svc, gbc, lightgbmc, xgboostc
stack_gen = StackingCVClassifier(classifiers=(ridgec, lr_elasticnet, svc, gbc, lightgbmc, xgboostc),
                                meta_classifier=xgboostc,
                                use_features_in_secondary=True)'''