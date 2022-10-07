#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (c) 2020, Isaac Boixaderas Coderch
                    Petar Radojkovic
                    Paul Carpenter
                    Marc Casas
                    Eduard Ayguade
                    Contact: isaac.boixaderas [at] bsc [dot] es
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


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
        'defined in the form of pandas time offset aliases, see '
        'https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases '
        'for more information.'
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('-pw', '--prediction-window', dest='pred_wind',
                        default='1d', help=help_aliases%'prediction window')
    parser.add_argument('--mitigation-time', dest='mitigation_cost',
                        default='2min', help=(help_aliases%'the time needed for '
                                             'performing an impact mitigation'))
    parser.add_argument('-i-preds', '--input-predictions', dest='preds_file',
                        default='data/predictions.csv',
                        help=('Path to the input predictions file, a CSV '
                              'containing the predictions data generated '
                              'by the train_test script.'))
    parser.add_argument('-i-ues', '--input-ues', dest='ues_file',
                        default='data/ues_reduction.csv',
                        help=('Path to the input Uncorrected Errors file, '
                              'a CSV containing the UEs data.'))
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