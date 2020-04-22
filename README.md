# UEPREDICT
Method for model prediction and evaluation of DRAM Uncorrected Errors (UEs).

 - The file [ue_predict/train_test.py](ue\_predict/train\_test.py) contains the code for applying cross-validation to the given features dataset and store the resulting predictions
 - The file [ue_precict/evaluation.py](ue\_predict/evaluation.py) contains the code for evaluating the predictions performed by the "train_test" file
 - The scripts [train_test.py](train\_test.py) and [evaluation.py](evaluation.py) provide a way of executing the afore-mentioned files by allowing to configure specific parameters, such as the length of the prediction window or the prediction frequency

The UEPREDICT code is released under the BSD-3 [License](LICENSE).
