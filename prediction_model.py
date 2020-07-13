import pandas as pd
import numpy as np
from model_setup_module import setup_and_train_model, predict_result
from import_data_module import import_data, create_test_train

#columns = ['age', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'ca', 'male', 
#'cp_typical_angina', 'cp_atypical_angina', 'cp_non_angina', 'ecg_st_t_abnorm', 'ecg_lvh',
#'thal_normal', 'thal_fixed_defect', 'thal_reversable']

datapoint = [26, 120, 250, 1, 160, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]

def initiate_setup():
    datafram = import_data()
    X_train, X_test, y_train, y_test = create_test_train(datafram)
    return setup_and_train_model(X_train, y_train, X_test, y_test)

def perform_prediction(model, datapoint):
    x_pred = np.array(datapoint).reshape(1,-1)
    prob_of_disease = predict_result(model, x_pred)[0,1]
    print("The probability of heart disease is {0:.2f}%".format(prob_of_disease * 100))

new_model = initiate_setup()
perform_prediction(new_model, datapoint)