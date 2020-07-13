from sklearn.linear_model import LogisticRegression
import logging

def get_model():
    return LogisticRegression()

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

def predict_result(model, df):
    return model.predict_proba(df)

def setup_and_train_model(X_train, y_train, x_test, y_test):
    model = get_model()
    train_model(model, X_train, y_train)
    logging.basicConfig(level=logging.INFO)
    logging.info("The training accuracy of the model: {0:.2f}%".format(model.score(X_train,y_train) * 100))
    logging.info("The test accuracy of the model: {0:.2f}%".format(model.score(x_test,y_test) * 100))
    return model
