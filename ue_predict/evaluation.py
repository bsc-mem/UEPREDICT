
"""
Module with all the necessary functions for performing the evaluation of the
Uncorrected Error predictions given by the train_test script.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from ue_predict.utils import *



def print_eval(pred_wind, threshold, tn, fp, fn, tp,
               mitigations, ues_predicted, ues_predictable, lead_time_maxs):
    """Print evaluation of the final predictions."""
    print(f'Prediction window: {pred_wind}',
        f'Decision threshold: {threshold}',
        f'Mitigations: {mitigations}',
        f'UEs predicted: {ues_predicted}',
        f'UEs predictable: {ues_predictable}',
        f'Lead times before UEs:',
        f'\tMaximum average: {np.mean(lead_time_maxs)}'
    , sep='\n')


def evaluate(df_pred, ues_df, pred_wind, mitigation_td,
             threshold=0.5, verbose=False):
    """Evaluate final predictions."""
    # confusion matrix
    tn, fp, fn, tp = ravel_conf_mat(confusion_matrix(
        df_pred['y_true'],
        (df_pred['y_prob_1'] > threshold).astype(int)
    ))
    
    lead_time_maxs = []
    ues_predicted = 0
    ues_predictable = 0
    
    date_times = df_pred.index.get_level_values('date_time')
    id_blades = df_pred.index.get_level_values('id_blade')
    # get positive predictions
    positives_df = df_pred[df_pred['y_prob_1'] > threshold]
    positive_dts = positives_df.index.get_level_values('date_time')
    positive_id_blades = positives_df.index.get_level_values('id_blade')
    
    for _, ue in ues_df.iterrows():
        # check if each of the UEs was predictable and/or predicted
        blade_preds_df = positives_df[
            (positive_id_blades == ue['id_blade']) &
            (ue['date_time'] - pred_wind <= positive_dts) &
            (positive_dts < ue['date_time'] - mitigation_td)
        ]
        if not blade_preds_df.empty:
            # predicted the node's going to have an UE
            blade_dts = blade_preds_df.index.get_level_values('date_time')
            time_diffs = ue['date_time'] - blade_dts
            lead_time_maxs.append(time_diffs.max())
            ues_predicted += 1
            ues_predictable += 1
        else:
            # filter UEs that were predictable, but not predicted
            blade_not_preds_df = df_pred[
                (id_blades == ue['id_blade']) &
                (ue['date_time'] - pred_wind <= date_times) &
                (date_times < ue['date_time'])
            ]
            if not blade_not_preds_df.empty:
                # predictable UE
                ues_predictable += 1
    
    # number of impact mitigations performed
    mitigations = df_pred[(df_pred['y_prob_1'] > threshold)] \
                        .groupby(level=['date_time', 'id_blade']) \
                        .size() \
                        .size
    
    # TODO print table for different thresholds
    # print evaluation
    if verbose:
        print_eval(pred_wind, threshold,tn, fp, fn, tp,
                   mitigations, ues_predicted, ues_predictable, lead_time_maxs)
    
    # TODO generate charts