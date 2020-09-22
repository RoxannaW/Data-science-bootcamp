
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import RepeatedKFold, KFold

def test_model(model, X_test, y_test, n_splits):
    """
    Function to show the score or mean squared error of the test set, divided in n_splits sections. 
    """
    n_splits_x_test = n_splits
    number = 1
    # val es un trozo que nunca se repite
    for (_, one_split) in KFold(n_splits=n_splits_x_test).split(X_test):
        y_pred = model.predict(X_test[one_split])
        print("-----------------------------")
        print("Mean squared error test", number, ": ", mean_squared_error(y_test[one_split], y_pred))
        number += 1