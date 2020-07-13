import pandas as pd
from sklearn.model_selection import train_test_split

filename = 'cleanData.csv'

def import_data():
    return pd.read_csv(filename)

def create_test_train(data, test_size = 0.3):
    x = data.drop(['target'], axis = 1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= test_size, random_state=123)
    return X_train, X_test, y_train, y_test
