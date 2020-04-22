#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performs evaluation of the predictions dataset given by the train_test module.
"""

import argparse
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

from ue_predict.evaluation import evaluate



def parse_args():
    # TODO change input file defaults and add helps
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
                        default='predictions.csv',
                        help=('Path to the input predictions file, a CSV '
                              'containing the predictions data generated '
                              'by the train_test script.'))
    parser.add_argument('-i-ues', '--input-ues', dest='ues_file',
                        default='../../data/ues_reduced_blade_1w.feather',
                        help=('Path to the input Uncorrected Errors file, '
                              'a CSV containing the UEs data.'))
    parser.add_argument('--mitigation-cost', dest='mitigation_cost',
                        default='2min', help=help_aliases%'mitigation cost')
    # TODO uncomment and implement
    # parser.add_argument('-o', '--output-evaluations', dest='evals_file',
    #                     default='evaluations.csv', help='TODO')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help=('If specified, shows the information regarding '
                              'the evaluation of the predictions.'))

    return parser.parse_args()


def main():
    # read and process arguments
    args = parse_args()
    pred_wind = pd.to_timedelta(args.pred_wind)
    mitigation_td = pd.to_timedelta(args.mitigation_cost)

    if args.verbose:
        print('Reading data...')
    identity = ['date_time', 'id_blade', 'dimm_id']
    preds_df = pd.read_csv(args.preds_file, parse_dates=['date_time']) \
                 .set_index(identity)
    # TODO parse datetimes + csv
    ues_blade_1w = pd.read_feather(args.ues_file)

    # evaluate model performance
    evaluate(
        preds_df, ues_blade_1w, pred_wind, mitigation_td,
        threshold=0.5, verbose=args.verbose
    )

    # TODO save evals


if __name__ == '__main__':
    main()