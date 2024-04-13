"""Training of the first model."""
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib
# local imports
from editor.data_manipulation import load_train_val_test_set
from editor.data_visualisation import (
    plot_calibration_curves,
    plot_ROC_PRC_conf,
    plot_train_feature_correlation,
    plot_model_feature_importances
)
TRAIN = False
work_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(work_dir)

# load consistent train-val-test splits for all trained models
train_X, val_X, _, train_y, val_y, _ = load_train_val_test_set(model='v1')

if TRAIN:
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    param_grid = {'max_depth': [10, 15, 20]}
    # define classifier
    model = GridSearchCV(
        RandomForestClassifier(
            oob_score=True, criterion='log_loss'
            ),
        param_grid=param_grid,
        cv=skf,
        scoring='roc_auc',
        refit=True,
        n_jobs=-1
        )
    model.fit(train_X, train_y)

    model = model.best_estimator_

    # save model
    if not os.path.exists('models'):
        os.mkdir('models')
    joblib.dump(model, 'models/model_v1.pkl')

# %% calibration curves for validation and train sets
model = joblib.load('models/model_v1.pkl')
train_prob_pred = pd.Series(
    model.predict_proba(train_X)[:, 1],
    index=train_X.index
    )
val_prob_pred = pd.Series(model.predict_proba(val_X)[:, 1], index=val_X.index)

plot_calibration_curves(train_prob_pred, val_prob_pred, train_y, val_y)

plot_ROC_PRC_conf(train_prob_pred, train_y, sup_tit='Training')

plot_ROC_PRC_conf(val_prob_pred, val_y, sup_tit='Validation')
