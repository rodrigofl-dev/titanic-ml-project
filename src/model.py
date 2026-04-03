SEED = 42

from preprocessing import TitanicAgeImputer

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Creating Pipelines
ada_pipeline = Pipeline(steps=[
    ('age_imputer', TitanicAgeImputer()),
    ('safety_imputer', SimpleImputer(strategy='median')),
    ('model', AdaBoostClassifier(
        algorithm='SAMME',
        estimator=DecisionTreeClassifier(max_depth=2, random_state=SEED),
        n_estimators=200,
        learning_rate=0.05,
        random_state=SEED
    ))
])

xgb_pipeline = Pipeline(steps=[
    ('age_imputer', TitanicAgeImputer()),
    ('safety_imputer', SimpleImputer(strategy='median')),
    ('model', XGBClassifier(
        eval_metric = 'logloss',
        colsample_bytree = 0.8,
        gamma = 0.5,
        learning_rate = 0.01,
        max_depth = 5,
        n_estimators = 100,
        subsample = 0.8,
        random_state=SEED
    ))
])

voting_ensemble = VotingClassifier(
    estimators=[
        ('AdaBoost', ada_pipeline),
        ('XGBoost', xgb_pipeline)
    ],
    voting='soft',
    n_jobs=-1
)