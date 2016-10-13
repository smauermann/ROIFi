import numpy as np
from sklearn.cross_validation import LeaveOneOut, cross_val_predict
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
# statsmodels.formula.api import ols
from weird import WEIRD
from sklearn.grid_search import GridSearchCV

def roi_classify(data, labels, n_subj, estimator='weird', scoring='recall',
                 covariates=None, normalizing=False, verbose=1):
    if estimator.lower() == 'weird':
        estimator = WEIRD()
    elif estimator.lower() == 'svc':
        estimator = GridSearchCV(estimator=SVC(kernel='linear'),
                                 param_grid=dict(C=[0.01, 0.1, 1, 10, 100, 1000, 10000]))

    if scoring == 'recall':
        scoring = recall_score

    if normalizing:
        data = normalize(data)

    pred = cross_val_predict(estimator, data, y=labels, cv=LeaveOneOut(n_subj))
    score = scoring(labels, pred, pos_label=None, average='macro')
    if verbose == 1:
        print('Recall: %.5f' % score)

    if covariates is not None:
        covstr = '+'.join(covariates.keys())
        data_cor = np.zeros_like(data)
        for i in range(data.shape[1]):
            covariates.update(y=data[:, i])
            # data_cor[:, i] = ols('y ~ %s' % covstr, covariates).fit().resid.values + np.mean(data[:, i])
        pred_cor = cross_val_predict(estimator, data_cor, y=labels, cv=LeaveOneOut(n_subj))
        score_cor = scoring(labels, pred_cor)
        print('Recall (corrected): %.5f' % score_cor)

    return score
