import numpy as np
import pandas as pd


def ravel_conf_mat(conf_mat):
    """Ravel confusion matrix."""
    cm = conf_mat.ravel()
    if len(cm) == 1:
        return cm[0], 0, 0, 0
    return cm


def print_conf_mat(conf_mat, sufix=''):
    """Print confusion matrix."""
    tn, fp, fn, tp = ravel_conf_mat(conf_mat)
    if sufix:
        sufix = f'({sufix})'
    print(f'TN:{tn}, FP:{fp}, FN:{fn}, TP:{tp} {sufix}')


def get_mitigations(df, threshold=0.5):
    """Return number of impact mitigations performed."""
    return df[(df['y_prob_1'] > threshold)] \
            .groupby(level=['date_time', 'id_blade']) \
            .size() \
            .size


def get_importances_df(ml_algo, ft_names):
    """Get feature importances given by the ML algorithm."""
    if not hasattr(ml_algo, 'feature_importances_'):
        raise ValueError('ML algorithm must have "feature_importances_"')
    
    return pd.DataFrame(
        ml_algo.feature_importances_,
        index=ft_names,
        columns=['importance']
    ).sort_values('importance', ascending=False)