
Example files containing the necessary information for running the scripts:

 - **ues\_reduction**: Uncorrected Errors information. It is assumed that the UEs have been processed as explained in Section II-C in the SC20 paper ["Cost-Aware Prediction of Uncorrected DRAM Errors in the Field"](https://dl.acm.org/doi/10.5555/3433701.3433782).
 - **features**: features for training the model. They are specified in Table I in the SC20 paper ["Cost-Aware Prediction of Uncorrected DRAM Errors in the Field"](https://dl.acm.org/doi/10.5555/3433701.3433782).
 - **predictions**: predictions performed by the model on the cross-validation method. It is the output file of the [train_test.py](../train\_test.py) script.
 - **evaluations**: evaluations of the predictions. It is the output file of the [evaluation.py](../evaluation.py) script.