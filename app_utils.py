import pandas as pd

def scale_dataset(dataset):
    dfSub = dataset - dataset.mean()
    return dfSub / dataset.std()