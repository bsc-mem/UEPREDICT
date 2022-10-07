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


def get_ues_predicted(df_pred, ues_df, pred_wind, mitigation_td, threshold=0.5):
    """Return the number of UEs that were correctly predicted."""
    ues_predicted = 0
    # get positive predictions
    positives_df = df_pred[df_pred['y_prob_1'] > threshold]
    positive_dts = positives_df.index.get_level_values('date_time')
    positive_id_blades = positives_df.index.get_level_values('id_blade')

    for _, ue in ues_df.iterrows():
        # check if each of the UEs was predictable and/or predicted
        blade_preds_df = positives_df[
            (positive_id_blades == ue['id_blade']) &
            (ue['date_time'] - pred_wind <= positive_dts) &
            # ignore the ones predicted without enough lead time
            (positive_dts < ue['date_time'] - mitigation_td) 
        ]
        if not blade_preds_df.empty:
            ues_predicted += 1

    return ues_predicted


def get_ues_predictable(df_pred, ues_df, pred_wind, mitigation_td):
    """Return the number of UEs that were predictable."""
    ues_predictable = 0
    # get positive instances
    positives_df = df_pred[df_pred['y_true'] == 1]
    positive_dts = positives_df.index.get_level_values('date_time')
    positive_id_blades = positives_df.index.get_level_values('id_blade')

    for _, ue in ues_df.iterrows():
        predictable_df = positives_df[
            (positive_id_blades == ue['id_blade']) &
            (ue['date_time'] - pred_wind <= positive_dts) &
            # ignore the ones predictable without enough lead time
            (positive_dts < ue['date_time'] - mitigation_td)
        ]
        if not predictable_df.empty:
            ues_predictable += 1

    return ues_predictable


def get_mitigations(df, threshold=0.5):
    """Return number of impact mitigations performed."""
    return df[(df['y_prob_1'] > threshold)] \
            .groupby(level=['date_time', 'id_blade']) \
            .size() \
            .size


def get_performance(df_pred, ues_df, pred_wind, mitigation_td, threshold=0.5):
    """
    Return the number of correctly predicted UEs (TP), total UEs predictable
    and number of impact mitigations performed (FP+TP)."""
    pred_dts = df_pred.index.get_level_values('date_time')
    dt_cond = ((pred_dts.min() <= ues_df['date_time']) &
               (ues_df['date_time'] <= pred_dts.max()))

    # number of correctly predicted UEs
    ues_predicted = get_ues_predicted(df_pred, ues_df[dt_cond], pred_wind,
                                      mitigation_td, threshold)
    # number of predictable UEs
    ues_predictable = get_ues_predictable(df_pred, ues_df[dt_cond], pred_wind,
                                          mitigation_td)
    # number of impact mitigations performed
    mitigations = get_mitigations(df_pred, threshold)

    return ues_predicted, ues_predictable, mitigations


def get_importances_df(ml_algo, ft_names):
    """Get feature importances given by the ML algorithm."""
    if not hasattr(ml_algo, 'feature_importances_'):
        raise ValueError('ML algorithm must have "feature_importances_"')
    
    return pd.DataFrame(
        ml_algo.feature_importances_,
        index=ft_names,
        columns=['importance']
    ).sort_values('importance', ascending=False)