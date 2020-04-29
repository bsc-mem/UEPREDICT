#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performs train/test cross-validation of the features dataset and allows to 
specify multiple options such as prediction window length,
predictions frequency, etc.
"""

import argparse
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

from ue_predict.train_test import get_target, train_test_cv



def parse_args():
    parser = argparse.ArgumentParser()
    help_aliases = (
        'Specifies the %s. It has to be '
        'defined in the form of pandas time offset aliases, see '
        'https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases '
        'for more information.'
    )
    parser.add_argument('-pw', '--prediction-window', dest='pred_wind',
                        default='1d', help=help_aliases%'prediction window')
    parser.add_argument('-pf', '--prediction-frequency', dest='pred_freq',
                        default='1min', help=help_aliases%'prediction frequency')
    parser.add_argument('-tf', '--train-frequency', dest='train_freq',
                        default='7d', help=help_aliases%'training frequency')
    parser.add_argument('-i-fts', '--input-features', dest='fts_file',
                        default='data/features.csv',
                        help=('Path to the input features file, a CSV '
                              'containing the features data.'))
    parser.add_argument('-i-ues', '--input-ues', dest='ues_file',
                        default='data/ues_reduction.csv',
                        help=('Path to the input Uncorrected Errors file, '
                              'a CSV containing the UEs data.'))
    parser.add_argument('-o', '--output-predictions', dest='preds_file',
                        default='data/predictions.csv',
                        help=('Path of the output file, a CSV containing '
                              'the predictions data after the train/test '
                              'iterations.'))
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help=('If specified, shows information during '
                              'the execution, such as performance, '
                              'training times... at each train/test split.'))

    return parser.parse_args()


def main():
    # read and process arguments
    args = parse_args()
    pred_wind = pd.to_timedelta(args.pred_wind)

    # read and process datasets
    if args.verbose:
        print('Reading data...')
    identity = ['date_time', 'id_blade', 'dimm_id']
    df = pd.read_csv(args.fts_file, parse_dates=['date_time'])
    ues_reduction = pd.read_csv(args.ues_file, parse_dates=['date_time'])

    # remove the last instances for which there is not enough time
    # for knowing the true outcome
    df = df[df['date_time'] < df['date_time'].max() - pred_wind]

    # recompute date_time based on the prediction frequency
    df = df.rename(columns={'date_time': 'real_date_time'})
    df['date_time'] = df['real_date_time'].dt.ceil(args.pred_freq)
    # take the last DIMM instance in each prediction interval
    df = df.groupby(['date_time', 'dimm_id'], as_index=False).last()
    df = df.set_index(identity)

    # compute target based on the real event datetime
    if args.verbose:
        print('Computing target...')
    target = get_target(df, ues_reduction, pred_wind)
    df['target'] = target
    df = df.drop('real_date_time', axis=1)

    # define basic parameters
    ml_algo = RandomForestClassifier
    hyperparams = {
        'n_estimators': [100],
        'max_depth': [None, 5, 10, 30, 80],
        'min_samples_split': [2, 10, 50, 100],
        'oob_score': [True],
        'random_state': [42],
        'n_jobs': [-1],
    }
    sampling_fn = RandomUnderSampler(random_state=42)
    start_at = None

    if start_at is None:
        # start the day before the first entry
        start_at = df.index.get_level_values('date_time').min().date()
    
    preds_df = train_test_cv(
        df, args.pred_freq, pred_wind, start_at, args.train_freq,
        ml_algo, hyperparams, sampling_fn, verbose=args.verbose
    )
    
    # save prediction probabilities
    preds_df.to_csv(args.preds_file)


if __name__ == '__main__':
    main()