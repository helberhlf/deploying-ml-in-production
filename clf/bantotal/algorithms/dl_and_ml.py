#-------------------------------------------------------
# @author Helber
#-------------------------------------------------------

# Importing library for manipulation and exploration of datasets.
import numpy as np
import pandas as pd

# Importing class to search for the ideal Hyperparameters selection.
from sklearn.model_selection import RandomizedSearchCV

# Importing libraries needed for predictive modeling, Machine Learning and Deep Learning algorithms..
import xgboost as xgb
#-------------------------------------------------------

## Machine Learning

# Creating a function to select the best features
def feature_imp(xFeatures, target,scoring,n_best_features, n_iter=None, cv=None, seed=None):
    # Define o classificador, ou seja, cria um objeto da classe XGBRFClassifier
    clf_XBGRFC = xgb.XGBRFClassifier()
    # Define the ideal Hyperparameters for creating the model
    params = {
        'objective'       :['binary:logistic'],
        'booster'         :['gbtree', 'dart'],
    #   'eval_metric'     :['auc'],
        "n_estimators"    :range(400, 900, 100),
        "max_depth"       :range(2, 12, 2),
        "learning_rate"   :np.linspace(1e-3, 0.5, 40),
        "gamma"           :np.linspace(1e-3, 2, 40),
        "reg_alpha"       :np.linspace(1e-3, 3, 40),
        "reg_lambda"      :np.linspace(1e-3, 3, 40),
    #   "min_child_weight":range(0,5,1),
        "subsample"       :[.5, .6, .7, .75, .8, .85, .9],
        "colsample_bytree":[.5, .6, .7, .75, .8, .85, .9,],
        "scale_pos_weight":[0],
        "importance_type" :['gain', 'cover', 'total_gain', 'total_cover'],
    }
    # Hyperparameter Tuning with RndCV
    clf_rdcv = RandomizedSearchCV(clf_XBGRFC, param_distributions=params,
                                  n_iter=n_iter, scoring=scoring, cv=cv,
                                  verbose=1, n_jobs=-1, random_state=seed
    )
    # Fit with HyperParameter tuning Random Search
    clf_rdcv.fit(xFeatures, target)
    print(f'Best Score: {clf_rdcv.best_score_}')
    print(f'\nBest params: \n{clf_rdcv.best_params_}')

    # Creating the model with the ideals estimators
    best_clf_rdcv = clf_rdcv.best_estimator_
    # Capturing importance of each feature
    feature_imp = best_clf_rdcv.get_booster().get_score()
    # Creating dataframe with what key features
    best_features = pd.DataFrame(data=feature_imp.values(), index=feature_imp.keys(),
                                 columns=["score_XGBRFClassifier"]).sort_values(by="score_XGBRFClassifier", ascending=True).nlargest(n_best_features, columns="score_XGBRFClassifier" )
    # Return the best features
    return best_features