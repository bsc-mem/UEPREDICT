
"""
Module with all the necessary functions for performing the evaluation of the
Uncorrected Error predictions given by the train_test script.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix



def print_eval(pred_wind, ue_costs, evals_df):
    """Print evaluation of the final predictions."""
    print(f'Prediction window: {pred_wind}')
    print(f'Total UEs predictable: {evals_df.iloc[0].ues_predictable}\n')
    cols_rename = {
        'threshold': 'Threshold',
        'mitigations': 'Mitigations',
        'ues_predicted': 'UEs predicted',
    }
    cols_rename.update({
        f'total_cost_ue_{ue_cost}': f'Total cost (UE {ue_cost})'
        for ue_cost in ue_costs
    })
    evals_str = evals_df \
                    .rename(columns=cols_rename) \
                    .set_index('Threshold') \
                    .drop(['lead_time_maxs_average', 'ues_predictable'], axis=1) \
                    .to_string()
    print(evals_str)


def get_total_cost(mitigations, predicted, n_ues, mitigation_cost, ue_cost):
    """Returns total cost of prediction method"""
    return mitigations*mitigation_cost + (n_ues-predicted)*ue_cost
    

def evaluate(df_pred, ues_df, pred_wind, mitigation_td, threshold=0.5):
    """Evaluate final predictions."""
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
    
    return pd.Series({
        'threshold': threshold,
        'mitigations': mitigations,
        'ues_predicted': ues_predicted,
        'ues_predictable': ues_predictable,
        'lead_time_maxs_average': np.mean(lead_time_maxs),
    })


def evaluate_multithreshold(df_pred, ues_df, pred_wind, mitigation_td,
                            verbose=False):
    # compute evaluations for multiple decision thresholds
    thresholds = np.arange(0, 1.1, 0.1)
    evals_df = pd.DataFrame([
        evaluate(df_pred, ues_df, pred_wind, mitigation_td, threshold=th)
        for th in thresholds
    ])
    # set floats to int
    cols2int = ['mitigations', 'ues_predicted', 'ues_predictable']
    evals_df[cols2int] = evals_df[cols2int].astype(int)

    # compute total cost for different UE costs (in server hours)
    ue_costs = [5, 50, 500]
    # set mitigation cost to hours
    mitigation_cost = mitigation_td.total_seconds()/3600
    for ue_cost in ue_costs:
        evals_df[f'total_cost_ue_{ue_cost}'] = evals_df.apply(
            lambda row: get_total_cost(
                mitigations=row['mitigations'],
                predicted=row['ues_predicted'],
                n_ues=len(ues_df),
                mitigation_cost=mitigation_cost,
                ue_cost=ue_cost
            ), axis=1
        )
    
    # print evaluations
    if verbose:
        print_eval(pred_wind, ue_costs, evals_df)

    # TODO generate charts

    return evals_df