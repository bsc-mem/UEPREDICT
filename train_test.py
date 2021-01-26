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
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler

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
    parser.add_argument('-clf', '--classifier', dest='classifier', default='RF',
                        choices=['RF', 'GBDT', 'LR', 'GNB', 'SVM', 'NN'],
                        help=('Classifier to use for training and predicting. ',
                              'Choices: RF, GBDT, LR, GNB, SVM or NN.'))
    parser.add_argument('-ru', '--random-undersampling', dest='under_ratio',
                        default=1, type=float,
                        help=('Ratio of the number of samples in the minority class ',
                              'over the number of samples in the majority class ',
                              'after resampling. The ratio is expressed as ',
                              'Nm/NrM, where Nm is the number of samples in ',
                              'the minority class and NrM is the number of ',
                              'samples in the majority class after resampling. ',
                              'If the ratio is 0, do not sample.'))
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help=('If specified, shows information during '
                              'the execution, such as performance, '
                              'training times... at each train/test split.'))

    return parser.parse_args()


def get_nn_model(input_dim):
    def get_model():
        """
        Return a keras model of a deep neural network."""
        nn_model = Sequential()
        nn_model.add(Dense(512, input_dim=input_dim, activation='relu'))
        nn_model.add(Dropout(0.4))
        nn_model.add(Dense(128, activation='relu'))
        nn_model.add(Dropout(0.4))
        nn_model.add(Dense(32, activation='relu'))
        nn_model.add(Dropout(0.4))
        nn_model.add(Dense(2, activation='softmax'))
        nn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                        metrics=['accuracy'])
        return nn_model
    return get_model


def get_classifier(opt, input_dim):
    """
    Return a tuple with the ML classifier to be used and its hyperparameter
    options (in dict format)."""
    if opt == 'RF':
        ml_algo = RandomForestClassifier
        hyperparams = {
            'n_estimators': [100],
            'max_depth': [None, 10, 30, 50, 100],
            'min_samples_split': [2, 10, 50, 100],
            'random_state': [42],
            'n_jobs': [-1],
        }
    elif opt == 'GBDT':
        ml_algo = LGBMClassifier
        hyperparams = {
            'boosting_type': ['gbdt'],
            'n_estimators': [100],
            'max_depth': [-1, 10, 30, 50, 100],
            'num_leaves': [2, 3, 5, 10, 50],
            'learning_rate': [0.001, 0.01, 0.1],
            'class_weight': [None, 'balanced'],
            'random_state': [42],
            'n_jobs': [-1],
        }
    elif opt == 'LR':
        ml_algo = LogisticRegression
        hyperparams = {
            'solver': ['newton-cg', 'lbfgs', 'saga'],
            'C': [0.0001, 0.001, 0.01],
            'class_weight': [None, 'balanced'],
            'random_state': [42],
            'n_jobs': [-1],
        }
    elif opt == 'GNB':
        ml_algo = GaussianNB
        hyperparams = {
            'var_smoothing': [10**-i for i in range(2, 15)],
        }
    elif opt == 'SVM':
        ml_algo = SVC
        hyperparams = {
            'probability': [True],
            'C': [0.01, 0.1, 1, 10],
            'gamma': [0.001, 0.01, 0.1, 1],
        }
    elif opt == 'NN':
        ml_algo = KerasClassifier(get_nn_model(input_dim), epochs=30, verbose=0)
        hyperparams = {}
    else:
        raise ValueError(f'{opt} is an invalid classifier name.')

    return ml_algo, hyperparams


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

    # scale features
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(df.drop('target', axis=1))
    scaled_df = pd.DataFrame(scaled_df, columns=df.drop('target', axis=1).columns, index=df.index)
    scaled_df['target'] = df['target']

    # get classifier with hyparameter options
    ml_algo, hyperparams = get_classifier(args.classifier, len(scaled_df.columns)-1)
    # sampling function
    sampling_fn = None
    if args.under_ratio > 0:
        sampling_fn = RandomUnderSampler(args.under_ratio, random_state=42)
    start_at = None
    if start_at is None:
        # start the day before the first entry
        start_at = scaled_df.index.get_level_values('date_time').min().date()
    
    preds_df = train_test_cv(
        scaled_df, args.pred_freq, pred_wind, start_at, args.train_freq,
        ml_algo, hyperparams, sampling_fn, ues_reduction, verbose=args.verbose
    )
    
    # save prediction probabilities
    preds_df.to_csv(args.preds_file)


if __name__ == '__main__':
    main()