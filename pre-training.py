import numpy as np
import os
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler


def f1(y_true, y_pred):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)

    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=4)

    tp = K.sum(y_true * y_pred, axis=0)
    tn = K.sum((1 - y_true) * (1 - y_pred), axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    return K.mean(2 * precision * recall / (precision + recall + K.epsilon()))


def categorical_focal_loss(gamma=2):
    """
        Categorical form of focal loss.
            FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
        References:
            https://arxiv.org/pdf/1708.02002.pdf
        Usage:
            model.compile(loss=categorical_focal_loss(gamma=2), optimizer="adam", metrics=["accuracy"])
            model.fit(class_weight={0:alpha0, 1:alpha1, ...}, ...)
        Notes:
           1. The alpha variable is the class_weight of keras.fit, so in implementation of the focal loss function
           we needn't define this variable.
           2. (important!!!) The output of the loss is the loss value of each training sample, not the total or average
            loss of each batch.
    """

    def focal_loss(y_true, y_pred):
        y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
        y_true = K.cast(y_true, y_pred.dtype)

        return K.sum(
            -y_true * K.pow(1 - y_pred, gamma) * K.log(y_pred + K.epsilon()), axis=-1)

    return focal_loss


def load_data(filename="dataset/mitdb.pkl"):
    import pickle

    with open(filename, "rb") as f:
        (x1_train, x2_train, y_train), (x1_test, x2_test, y_test) = pickle.load(f)

    return (x1_train, x2_train, y_train), (x1_test, x2_test, y_test)


if __name__ == "__main__":
    (x1_train, x2_train, y_train), (x1_test, x2_test, y_test) = load_data()

    x1_train = np.expand_dims(x1_train, axis=-1)
    x1_test = np.expand_dims(x1_test, axis=-1)

    scaler = RobustScaler()
    x2_train = scaler.fit_transform(x2_train)
    x2_test = scaler.transform(x2_test)

    model = load_model(os.path.join("models", "final_model.h5"),
                       custom_objects={"focal_loss": categorical_focal_loss(gamma=5),
                                       "f1": f1})
    model.summary()

    print("training:")
    y_true, y_pred = y_train, np.argmax(model.predict([x1_train, x2_train], batch_size=1024, verbose=1), axis=-1)

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=3))

    print("testing:")
    y_true, y_pred = y_test, np.argmax(model.predict([x1_test, x2_test], batch_size=1024, verbose=1), axis=-1)

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=3))
