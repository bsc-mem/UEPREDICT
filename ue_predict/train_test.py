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
Module with all the necessary function for performing train/test
cross-validation for predicting the Uncorrected Errors.
"""

import time
import itertools
import numpy as np
import pandas as pd
import datetime as dt

from sklearn.metrics import confusion_matrix

from ue_predict.utils import *



def get_target(df, ues_df, pred_wind):
    """Get target feature."""
    target = pd.Series(np.zeros(len(df)))
    id_blades = df.index.get_level_values('id_blade')
    for _, ue in ues_df.iterrows():
        # we're interested in events previous to the UE not at
        # the exact UE time(we have UE events in that moment)
        future_ue = ((df['real_date_time'] >= ue['date_time'] - pred_wind) &
                     (df['real_date_time'] < ue['date_time'])).values
        target.loc[(id_blades == ue['id_blade']) & future_ue] = 1
    return target.values


def sample(X, y, sampling_fn):
    """Sample the given X and y data"""
    if sampling_fn is None:
        return X, y
    
    if not hasattr(sampling_fn, 'fit_resample'):
        raise ValueError(('Sampling function must implement'
                          ' a "fit_resample" method'))

    X_samp, y_samp = sampling_fn.fit_resample(X, y)
    idxs = sampling_fn.sample_indices_
    X_samp = X_samp.set_index(X.iloc[idxs].index) \
                   .sort_index(axis=0, level='date_time')
    y_samp.index = y.iloc[idxs].index
    y_samp = y_samp.sort_index(axis=0, level='date_time')
    return X_samp, y_samp

    
def print_performance(train_preds_df, test_preds_df, ues_df,
                      pred_wind, mitigation_td):
    """Print model performance on train and test sets."""
    # train performance
    performance = get_performance(train_preds_df, ues_df, pred_wind, mitigation_td)
    ues_predicted, ues_predictable, mitigations = performance
    print(f'UEs predicted: {ues_predicted}, UEs predictable: {ues_predictable}, '
          f'Mitigations: {mitigations} (train)')
    # test performance
    performance = get_performance(test_preds_df, ues_df, pred_wind, mitigation_td)
    ues_predicted, ues_predictable, mitigations = performance
    print(f'UEs predicted: {ues_predicted}, UEs predictable: {ues_predictable}, '
          f'Mitigations: {mitigations} (test)')


def enough_data(train_data, test_data, verbose=False):
    """Check if train and test sets have any elements."""
    if train_data.empty:
        if verbose:
            print('Empty training data\n')
        return False
    if test_data.empty:
        if verbose:
            print('Empty testing data\n')
        return False
    return True


def enough_classes(y, verbose=False):
    """Check if target feature has enough classes."""
    if y.nunique() == 1:
        # continue iterating if only 1 class in target
        if verbose:
            print('Only 1 class for training\n')
        return False
    return True


def get_hyperparams_combinations(hyperparams):
    """Get list of hyperparmeter (dict) combinations."""
    # transforms tuning hyperparams to a list of dict params for each option
    return [
        {k:v for k,v in zip(hyperparams.keys(), hypms)}
         for hypms
         in itertools.product(*[vals for vals in hyperparams.values()])
    ]


def get_train_test_data(df, d, pred_wind, train_freq, ft_names, verbose):
    """Get train and test datasets."""
    train_limit = d - pred_wind
    test_limit = d + pd.to_timedelta(train_freq)

    if verbose:
        print(f'Train data < {train_limit}')
        print(f'Test data [{d}, {test_limit})')
        print('----------------------------')

    # TODO threading
    date_times = df.index.get_level_values('date_time').date
    train_idxs = df[date_times < train_limit].index
    test_idxs = df[(date_times >= d) & (date_times < test_limit)].index

    if not enough_data(train_idxs, test_idxs, verbose):
        # TODO improve return, it's ugly
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()

    # compute train and test sets
    train = df.loc[train_idxs]
    test = df.loc[test_idxs]
    X_train, y_train = train[ft_names], train['target']
    X_test, y_test = test[ft_names], test['target']
    
    return X_train, y_train, X_test, y_test


def hyperparamters_tuning(
        algo, tun_hyperparams, eval_hyperparams,
        X_train, y_train, X_test, y_test,
        verbose=False
    ):
    """Perform hyperparameters tuning."""
    
    if verbose:
        print('Tuning hyperparameters...')
    
    if X_train.size == 0:
        raise ValueError('Empty training set for' 
                         ' hyperparameters optimization')
    
    new_evals = eval_hyperparams[:]
    if not new_evals:
         new_evals = [[] for _ in range(len(tun_hyperparams))]
    
    # add evaluations for this hyperparams tuning iteration
    for i, hyperparams in enumerate(tun_hyperparams):
        model = algo(**hyperparams).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        tn, fp, fn, tp = ravel_conf_mat(confusion_matrix(y_test, y_pred))
        # cost approximations for optimizing hyperparameters
        mitigation_cost = 2/60
        ue_cost = 500
        new_evals[i].append((fp+tp)*mitigation_cost + fn*ue_cost)
    
    # index of the element that minimizes the mean evaluation
    idx_min = np.argmin(np.mean(new_evals, axis=1))
    
    if verbose:
        print(f'Best hyperparameters: {tun_hyperparams[idx_min]}')
    
    # return non-trained model for the best hyperparameters
    return algo(**tun_hyperparams[idx_min]), new_evals


def train_test_iteration(model, sampling_fn, ft_names,
                         X_train, y_train, X_test, y_test, verbose=False):
    """Perform an iteration of training and testing."""
    # training
    t_train1 = time.time()
    model.fit(X_train, y_train)
    t_train2 = time.time()

    # prediction
    t_pred1 = time.time()
    y_probs = model.predict_proba(X_test)
    t_pred2 = time.time()
    
    if verbose:
        print('Training time: %.2f secs' % (t_train2 - t_train1))
        print(f'Prediction time: %.2f secs' % (t_pred2 - t_pred1))

    # generate a DF with the predictions
    preds_df = pd.DataFrame({
        'y_true': y_test,
        'y_prob_1': y_probs[:,1] # probability of class 1
    })
    # compute feature importances
    ft_importances = []
    if hasattr(model, 'feature_importances_'):
        ft_importances = get_importances_df(model, ft_names)
    
    return preds_df, ft_importances


def train_test_cv(df, pred_freq, pred_wind, start_at_date, train_freq,
                  ml_algo, hyperparams, sampling_fn, ues_bdf, verbose=False):
    """
    Logic for applying cross-validation for training and testing.
    It also performs hyperparameters optimization."""
    # test set predictions DF
    test_preds_df = pd.DataFrame()
    # feature importances list (one element per test iteration)
    ft_importances = []

    if hyperparams:
        # hyperparams list of dictionaries for each hyperparam configuration
        tun_hyperparams = get_hyperparams_combinations(hyperparams)
        eval_hyperparams = []
    
    # feature names: remove target feature from predictors
    ft_names = np.delete(
        df.columns,
        np.where(df.columns == 'target')
    )
    
    # train/test dates list
    date_times = df.index.get_level_values('date_time')
    split_dates = pd.date_range(
        start_at_date,
        date_times.max().date() + pd.Timedelta(days=1),
        freq=train_freq
    ).date

    # cross-validation loop
    for d in split_dates:
        Xy_sets = get_train_test_data(df, d, pred_wind,
                                      train_freq, ft_names, verbose)
        X_train, y_train, X_test, y_test = Xy_sets
        
        no_classes = not enough_classes(y_train, verbose)
        if X_train.empty or no_classes:
            continue
        
        # sample training data
        X_train, y_train = sample(X_train, y_train, sampling_fn)
        
        if not hyperparams:
            model = ml_algo() if callable(ml_algo) else ml_algo
        elif not eval_hyperparams:
            # run hyperparams tuning if it's the first iteration
            model, eval_hyperparams = hyperparamters_tuning(
                ml_algo, tun_hyperparams, eval_hyperparams,
                X_train, y_train, X_test, y_test, verbose
            )
            continue
        
        # execute train/test
        preds_df, ft_imps = train_test_iteration(
            model, sampling_fn, ft_names,
            X_train, y_train, X_test, y_test, verbose
        )
        
        # update test_preds_df
        test_preds_df = pd.concat([test_preds_df, preds_df], sort=True)
        # feature importances
        ft_importances.append(ft_imps)
        
        if verbose:
            # print model performance on train and test sets
            train_preds_df = pd.DataFrame({
                'y_true': y_train,
                'y_prob_1': model.predict_proba(X_train)[:,1] # probability of class 1
            })
            # use a mitigation time of 2 minutes by default
            print_performance(train_preds_df, preds_df, ues_bdf,
                              pred_wind, mitigation_td=dt.timedelta(minutes=2))
        
        if hyperparams:
            # hyperparameters optimization
            t_hyper1 = time.time()
            model, eval_hyperparams = hyperparamters_tuning(
                ml_algo, tun_hyperparams, eval_hyperparams,
                X_train, y_train, X_test, y_test, verbose=verbose
            )
            t_hyper2 = time.time()
        
            if verbose:
                print('Hyperparameters optimization '
                      'time: %.2f secs' % (t_hyper2 - t_hyper1))
                print()
    
    return test_preds_df