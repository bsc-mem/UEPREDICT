#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performs evaluation of the predictions dataset given by the train_test module.
"""

import argparse
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

from ue_predict.evaluation import evaluate_multithreshold



def parse_args():
    help_aliases = (
        'Specifies the %s. It has to be '
        'specified in the form of pandas time offset aliases, see '
        'https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases '
        'for more information.'
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('-pw', '--prediction-window', dest='pred_wind',
                        default='1d', help=help_aliases%'prediction window')
    parser.add_argument('-i-preds', '--input-predictions', dest='preds_file',
                        default='data/predictions.csv',
                        help=('Path to the input predictions file, a CSV '
                              'containing the predictions data generated '
                              'by the train_test script.'))
    parser.add_argument('-i-ues', '--input-ues', dest='ues_file',
                        default='data/ues_reduction.csv',
                        help=('Path to the input Uncorrected Errors file, '
                              'a CSV containing the UEs data.'))
    parser.add_argument('--mitigation-cost', dest='mitigation_cost',
                        default='2min', help=help_aliases%'mitigation cost')
    parser.add_argument('-o', '--output-evaluations', dest='evals_file',
                        default='data/evaluations.csv',
                        help=('Path of the output file, a CSV containing '
                              'the evaluations data.'))
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help=('If specified, shows the information regarding '
                              'the evaluation of the predictions.'))

    return parser.parse_args()


def main():
    # read and process arguments
    args = parse_args()
    pred_wind = pd.to_timedelta(args.pred_wind)
    mitigation_td = pd.to_timedelta(args.mitigation_cost)

    # read and process datasets
    if args.verbose:
        print('Reading data...')
    identity = ['date_time', 'id_blade', 'dimm_id']
    preds_df = pd.read_csv(args.preds_file, parse_dates=['date_time']) \
                 .set_index(identity)
    ues_reduction = pd.read_csv(args.ues_file, parse_dates=['date_time'])

    # evaluate model performance
    evals_df = evaluate_multithreshold(
        preds_df, ues_reduction, pred_wind, mitigation_td, verbose=args.verbose
    )

    # save evals
    evals_df.to_csv(args.evals_file, index=False)


if __name__ == '__main__':
    main()