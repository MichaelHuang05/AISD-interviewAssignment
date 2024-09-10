# AISD-interviewAssignment
Interview assignment from AID team at FIT-Foxconn <br>
*** Since the `train.csv` and `test.csv` are sensitive data, it is not included in this repository. Please complete the repository with `train.csv` and `test.csv` by yourself.

# Output
The outputs are: 
1. Excel file with 3 sheets including predicted outcome(that is Exited) for `test.csv` and the predictive ability using 20% of test dataset in `train.csv`.
2. The ROC curve of the model over `train.csv` saved with `.png` file.

# Run the script
There are two model: `(1)xgboost` `(2)DNN` in this assignment. To get the output, please run shell script with command 
```python main.py --models xgboost,DNN``` 
to get outcomes for both models. The `--models` parameter decide which model to be run. With default model set to `xgboost`, ```python main.py``` will only get outcome for xgboost model.

# Modification 
#### 1. To include more models
- If one wants to include more models for comparison, please add it into the ```Train_model``` and ```make_prediction``` functions of the `ml_module.py` file.

#### 2. To use more epochs to train the DNN or n_estimators to train xgboost
- One can adjust the epoch or n_estimators by adding `num_epochs = 200` for DNN and `n_estimators = 1000` for xgboost into ```Train_model``` function in the `main.py` file.

#### 3. Adjust threshold for DNN
- Since the output for DNN is probability, one can affect the classification outcome through adjusting threshold. The default threshold here is `0.5`. That is, if predicted `Exited` probability from DNN is not smaller than `0.5`, it will be classified as `1`.
If someone wants to change the threshold, add the parameters `DNN_threshold = .5` into the function ```make_prediction``` in the `main.py` file.


