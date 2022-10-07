# UEPREDICT
Method for model prediction and evaluation of DRAM Uncorrected Errors (UEs).

Supplemental code for the SC20 paper ["Cost-Aware Prediction of Uncorrected DRAM Errors in the Field"](https://upcommons.upc.edu/bitstream/handle/2117/341921/UE-Prediction_print.pdf?sequence=1&isAllowed=y).

The project is structured as follows:

 - The folder [ue_predict](ue\_predict) is a module containing the necessary files and methods for training and evaluation.
 - The scripts [train_test.py](train\_test.py) and [evaluation.py](evaluation.py) provide a way of executing the 'ue_predict' module by allowing the configuration of specific parameters, such as the length of the prediction window or the prediction frequency.
 - The folder [data](data) contains the files needed for training and evaluation.
 - The folder [synthetic_ues](synthetic_ues) contains a synthetic log file of UEs based on the production MareNostrum 3 error logs. We provide this log file in order to enable future studies to quantify the real-world impact of DRAM uncorrected errors and any proposed resiliency techniques.

The UEPREDICT code is released under the BSD-3 [License](LICENSE).


## Running scripts

Scripts can be executed directly from the shell as python 3 files. First, execute the [train_test.py](train\_test.py) in order to compute the predictions using walk-forward validation. Scripts' arguments are discribed in next section.

```shell
python3 train_test.py --verbose
```

If executed with the `verbose` argument, it prints information at each training/testing step, such as the confusion matrix values for train and test sets or the best hyperparamters.

After its execution, it will generate a file with the probabilities calculated by the model of each instance belonging to class 1 (i.e. having an Uncorrected Error in the next prediction window), alongside the correct class label. This file is stored as [data/predictions.csv](data/predictions.csv) by default.

Once the `predictions` file is generated, it can be evaluated by executing the [evaluation.py](evaluation.py) script, which gives information such as the number of impact mitigations performed or the number of correctly predicted UEs.

```shell
python3 evaluation.py --verbose
```


## Arguments

### Train/test

| Switch | Long switch               | Description                                                                                                         |
| ------ | ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| -pw    | --prediction-window       | Specifies the prediction window. It has to be defined in the form of pandas time offset aliases, see [offset aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases) for more information.                                          |
| -pf    | --prediction-frequency    | Specifies the prediction frequency. It has to be defined in the form of pandas time offset aliases, see [offset aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases) for more information.                                  |
| -tf    | --train-frequency         | Specifies the training frequency. It has to be defined in the form of pandas time offset aliases, see [offset aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases) for more information.                                  |
| -i-fts | --input-features          | Path to the input features file, a CSV containing the features data.                                                |
| -i-ues | --input-ues               | Path to the input Uncorrected Errors file, a CSV containing the UEs data.                                           |
| -o     | --output-predictions      | Path of the output file, a CSV containing the predictions data after the train/test iterations.                     |
| -clf   | --classifier              | Classifier to use for training and predicting. Choices: RF, GBDT, LR, GNB, SVM or NN.                               |
| -ru    | --random-undersampling    | Ratio of the number of samples in the minority class over the number of samples in the majority class after resampling. The ratio is expressed as *N<sub>m</sub>/N<sub>rM</sub>*, where *N<sub>m</sub>* is the number of samples in the minority class and *N<sub>rM</sub>* is the number of samples in the majority class after resampling. If the ratio is 0, do not sample.                                                                                     |
|        | --verbose                 | If specified, shows information during the execution, such as performance, training times, etc. at each train/test split.                                                                                                                                                     |


### Evaluation

| Switch   | Long switch             | Description                                                                                                         |
| -------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------- |
| -pw      | --prediction-window     | Specifies the prediction window. It has to be defined in the form of pandas time offset aliases, see [offset aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases) for more information.                                          |
|          | --mitigation-time       | Specifies the time needed for performing an impact mitigation. It has to be defined in the form of pandas time offset aliases, see [offset aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases) for more information.                                                                                                                                               |
| -i-preds | --input-predictions     | Path to the input predictions file, a CSV containing the predictions data generated by the train_test script.       |
| -i-ues | --input-ues               | Path to the input Uncorrected Errors file, a CSV containing the UEs data.                                           |
| -o     | --output-evaluations      | Path of the output file, a CSV containing the evaluations data.                                                     |
|        | --verbose                 | If specified, shows the information regarding split the evaluation of the predictions.                              |


# Acknowledgment

This work was supported by the Spanish Ministry of Science and Technology (project PID2019-107255GB), Generalitat de Catalunya (contracts 2014-SGR-1051 and 2014-SGR-1272)
and the European Union’s Horizon 2020 research and innovation programme and EuroEXA project (grant agreement No 754337). Paul Carpenter and Marc Casas hold the Ramon
y Cajal fellowship under contracts RYC2018-025628-I and RYC2017-23269, respectively, of the Ministry of Economy and Competitiveness of Spain.