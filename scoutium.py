# The dataset used here is from Scoutium, a digital football monitoring platform.
# According to the characteristics of the football players observed in the matches, the football players are evaluated by the scouts.
# The dataset consists of information about football players' features and their scores.

import warnings
from sklearn.exceptions import ConvergenceWarning
import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

df1 = pd.read_csv("SCOUTIUM/scoutium_attributes.csv", sep=";")

### df1.columns ###

# task_response_id: The set of a scout's assessments of all players on a team's roster in a match
# match_id: The id of the match
# evaluator_id: The id of the scout
# player_id: The id of the player
# position_id: The id of the position played by the relevant player in that match
# 1: Keeper
# 2: Stopper
# 3: Right-back
# 4: Left back
# 5: Defensive midfielder
# 6: Central midfielder
# 7: Right wing
# 8: Left wing
# 9: Offensive midfielder
# 10: Striker
# analysis_id: The set of attribute evaluations of a scout for a player in a match
# attribute_id: The id of each attribute that the players were evaluated for
# attribute_value: The value (points) a scout gives to a player's attribute

df2 = pd.read_csv("SCOUTIUM/scoutium_potential_labels.csv", sep=";")

### df2.columns ###

# task_response_id: The set of a scout's assessments of all players on a team's roster in a match
# match_id: The id of the match
# evaluator_id: The id of the scout
# player_id: The id of the player
# potential_label: The label indicating the final decision of a scout regarding a player in a match (target variable)

df_=pd.merge(df1, df2, on=['task_response_id', 'match_id', 'evaluator_id', 'player_id'], how="left")
df=df_.copy()
df.head()

# Extract the keepers

df=df.loc[~(df["position_id"]==1)]

# Extract "below_average" label

df=df.loc[~(df["potential_label"]=="below_average")]

# Create pivot_table

attribute_value_table=pd.pivot_table(df, values="attribute_value", index= ["player_id","position_id", "potential_label"], columns=["attribute_id"])
attribute_value_table.reset_index(inplace=True)
attribute_value_table.columns.dtype
attribute_value_table.columns=attribute_value_table.columns.astype("str")
attribute_value_table.columns.dtype
attribute_value_table.info()

#LabelEncoding

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

label_encoder(attribute_value_table, "potential_label")

#cat_cols and num_cols

num_cols = [col for col in attribute_value_table.columns if (attribute_value_table[col].nunique()>7) and ("player_id" not in col)]

# StandardScaler

ss = StandardScaler()
X_scaled = ss.fit_transform(attribute_value_table[num_cols])
attribute_value_table[num_cols] = pd.DataFrame(X_scaled, columns=attribute_value_table[num_cols].columns)

#Base Model
def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

y = attribute_value_table["potential_label"]
X = attribute_value_table.drop(["potential_label"], axis=1)

base_models(X, y, scoring="roc_auc")

"""
roc_auc: 0.5548 (LR)
roc_auc: 0.5 (KNN)
roc_auc: 0.3674 (SVC)
roc_auc: 0.7152 (CART)
roc_auc: 0.8864 (RF)
roc_auc: 0.7788 (Adaboost)
roc_auc: 0.8264 (GBM)
roc_auc: 0.8308 (XGBoost)
roc_auc: 0.8493 (LightGBM)
"""

base_models(X, y, scoring="f1")

"""
f1: 0.0 (LR) 
f1: 0.184 (KNN) 
f1: 0.0 (SVC) 
f1: 0.5558 (CART) 
f1: 0.586 (RF) 
f1: 0.3098 (Adaboost) 
f1: 0.5135 (GBM) 
f1: 0.548 (XGBoost) 
f1: 0.5586 (LightGBM) 
"""

base_models(X, y, scoring="precision")

"""
precision: 0.0 (LR) 
precision: 0.1318 (KNN) 
precision: 0.0 (SVC) 
precision: 0.4506 (CART) 
precision: 0.8 (RF) 
precision: 0.3011 (Adaboost) 
precision: 0.5683 (GBM) 
precision: 0.5839 (XGBoost) 
precision: 0.6704 (LightGBM) 
"""

base_models(X, y, scoring="recall")

"""
recall: 0.0 (LR) 
recall: 0.4074 (KNN) 
recall: 0.0 (SVC) 
recall: 0.4298 (CART) 
recall: 0.5156 (RF) 
recall: 0.5 (Adaboost) 
recall: 0.5692 (GBM) 
recall: 0.5331 (XGBoost) 
recall: 0.5166 (LightGBM) 
"""

base_models(X, y, scoring="accuracy")

"""
accuracy: 0.7934 (LR) 
accuracy: 0.5547 (KNN) 
accuracy: 0.7934 (SVC) 
accuracy: 0.8154 (CART) 
accuracy: 0.8597 (RF) 
accuracy: 0.6214 (Adaboost) 
accuracy: 0.7534 (GBM) 
accuracy: 0.8193 (XGBoost) 
accuracy: 0.8268 (LightGBM) 
"""

# Automated Hyperparameter Optimization

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}


classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]


def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

#best_models = hyperparameter_optimization(X, y)

"""
########## KNN ##########
roc_auc (Before): 0.5
roc_auc (After): 0.5
KNN best params: {'n_neighbors': 8}

########## CART ##########
roc_auc (Before): 0.7095
roc_auc (After): 0.7236
CART best params: {'max_depth': 1, 'min_samples_split': 2}

########## RF ##########
roc_auc (Before): 0.8724
roc_auc (After): 0.8832
RF best params: {'max_depth': 15, 'max_features': 5, 'min_samples_split': 20, 'n_estimators': 200}

########## XGBoost ##########
roc_auc (Before): 0.8308
roc_auc (After): 0.8591
XGBoost best params: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100}

########## LightGBM ##########
roc_auc (Before): 0.8493
roc_auc (After): 0.8381
LightGBM best params: {'learning_rate': 0.01, 'n_estimators': 300}
"""

#hyperparameter_optimization(X, y, cv=3, scoring="f1")

"""
########## KNN ##########
f1 (Before): 0.184
f1 (After): 0.0
KNN best params: {'n_neighbors': 8}

########## CART ##########
f1 (Before): 0.4308
f1 (After): 0.5913
CART best params: {'max_depth': 1, 'min_samples_split': 2}

########## RF ##########
f1 (Before): 0.5679
f1 (After): 0.5835
RF best params: {'max_depth': None, 'max_features': 5, 'min_samples_split': 20, 'n_estimators': 300}

########## XGBoost ##########
f1 (Before): 0.548
f1 (After): 0.6159
XGBoost best params: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100}

########## LightGBM ##########
f1 (Before): 0.5586
f1 (After): 0.6075
LightGBM best params: {'learning_rate': 0.01, 'n_estimators': 300}
"""

#hyperparameter_optimization(X, y, cv=3, scoring="precision")

"""
########## KNN ##########
precision (Before): 0.1318
precision (After): 0.0
KNN best params: {'n_neighbors': 8}
########## CART ##########
precision (Before): 0.4931
precision (After): 0.9
CART best params: {'max_depth': 1, 'min_samples_split': 2}
########## RF ##########
precision (Before): 0.7546
precision (After): 0.8933
RF best params: {'max_depth': 8, 'max_features': 5, 'min_samples_split': 20, 'n_estimators': 200}
########## XGBoost ##########
precision (Before): 0.5839
precision (After): 0.7298
XGBoost best params: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100}
########## LightGBM ##########
precision (Before): 0.6704
precision (After): 0.7488
LightGBM best params: {'learning_rate': 0.01, 'n_estimators': 300}
"""

#hyperparameter_optimization(X, y, cv=3, scoring="recall")

"""
########## KNN ##########
recall (Before): 0.4074
recall (After): 0.0
KNN best params: {'n_neighbors': 8}
########## CART ##########
recall (Before): 0.3421
recall (After): 0.461
CART best params: {'max_depth': 1, 'min_samples_split': 2}
########## RF ##########
recall (Before): 0.5331
recall (After): 0.5136
RF best params: {'max_depth': 8, 'max_features': 5, 'min_samples_split': 15, 'n_estimators': 300}
########## XGBoost ##########
recall (Before): 0.5331
recall (After): 0.5507
XGBoost best params: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100}
########## LightGBM ##########
recall (Before): 0.5166
recall (After): 0.5341
LightGBM best params: {'learning_rate': 0.01, 'n_estimators': 300}
"""

hyperparameter_optimization(X, y, cv=3, scoring="accuracy")

"""
########## KNN ##########
accuracy (Before): 0.5547
accuracy (After): 0.7934
KNN best params: {'n_neighbors': 8}
########## CART ##########
accuracy (Before): 0.834
accuracy (After): 0.8781
CART best params: {'max_depth': 1, 'min_samples_split': 2}
########## RF ##########
accuracy (Before): 0.8488
accuracy (After): 0.8708
RF best params: {'max_depth': None, 'max_features': 5, 'min_samples_split': 15, 'n_estimators': 200}
########## XGBoost ##########
accuracy (Before): 0.8193
accuracy (After): 0.8633
XGBoost best params: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100}
########## LightGBM ##########
accuracy (Before): 0.8268
accuracy (After): 0.8598
LightGBM best params: {'learning_rate': 0.01, 'n_estimators': 300}
"""