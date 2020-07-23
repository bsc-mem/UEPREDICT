
"""
Module with all the necessary functions for performing the evaluation of the
Uncorrected Error predictions given by the train_test script.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from ue_predict.utils import *



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
                    .drop('ues_predictable', axis=1) \
                    .to_string()
    print(evals_str)


def get_total_cost(mitigations, predicted, n_ues, mitigation_cost, ue_cost):
    """Returns total cost of prediction method"""
    return mitigations*mitigation_cost + (n_ues-predicted)*ue_cost
    

def evaluate(df_pred, ues_df, pred_wind, mitigation_td, threshold=0.5):
    """Evaluate final predictions."""
    # get UEs predicted, UEs predictable and mitigations performed
    performance = get_performance(df_pred, ues_df, pred_wind, mitigation_td, threshold)
    
    return pd.Series({
        'threshold': threshold,
        'ues_predicted': performance[0],
        'ues_predictable': performance[1],
        'mitigations': performance[2],
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