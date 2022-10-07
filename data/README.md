
Example files containing the necessary information for running the scripts:

 - **ues\_reduction**: Uncorrected Errors information. It is assumed that the UEs have been processed as explained in Section II-C in the SC20 paper ["Cost-Aware Prediction of Uncorrected DRAM Errors in the Field"](https://upcommons.upc.edu/bitstream/handle/2117/341921/UE-Prediction_print.pdf?sequence=1&isAllowed=y).
 - **features**: features for training the model. They are specified in Table I in the SC20 paper ["Cost-Aware Prediction of Uncorrected DRAM Errors in the Field"](https://upcommons.upc.edu/bitstream/handle/2117/341921/UE-Prediction_print.pdf?sequence=1&isAllowed=y).
 - **predictions**: predictions performed by the model on the cross-validation method. It is the output file of the [train_test.py](../train\_test.py) script.
 - **evaluations**: evaluations of the predictions. It is the output file of the [evaluation.py](../evaluation.py) script.