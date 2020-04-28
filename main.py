from datetime import datetime

import keras
import numpy as np
import os
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard
from keras.layers import Activation, BatchNormalization, Concatenate, Conv1D, Dense, Flatten, Input, MaxPooling1D
from keras.models import Model
from keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(0)


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

        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        return K.sum(-y_true * K.pow(1 - y_pred, gamma) * K.log(y_pred), axis=-1)

    return focal_loss


def create_model(l=0.0):
    inputs1 = Input(shape=(200, 1))
    x1 = inputs1

    x1 = Conv1D(16, kernel_size=11, strides=3, kernel_initializer="he_normal", kernel_regularizer=l2(l),
                bias_regularizer=l2(l))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)

    x1 = MaxPooling1D(3, strides=2)(x1)

    x1 = Conv1D(32, kernel_size=5, kernel_initializer="he_normal", kernel_regularizer=l2(l),
                bias_regularizer=l2(l))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)

    x1 = MaxPooling1D(3, strides=2)(x1)

    x1 = Conv1D(64, kernel_size=3, kernel_initializer="he_normal", kernel_regularizer=l2(l),
                bias_regularizer=l2(l))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)

    x1 = MaxPooling1D(3, strides=2)(x1)

    x1 = Flatten()(x1)

    inputs2 = Input(shape=(4,))
    x2 = inputs2

    x = Concatenate()([x1, x2])

    x = Dense(64, kernel_initializer="he_normal", kernel_regularizer=l2(l), bias_regularizer=l2(l),
              activation="relu")(x)

    outputs = Dense(4, activation="softmax")(x)

    model = Model(inputs=(inputs1, inputs2), outputs=outputs)

    return model


class MyGenerator(keras.utils.Sequence):

    def __init__(self, x1, x2, y, batch_size):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.batch_size = batch_size
        self.indices = np.arange(len(self.x1))
        self.on_epoch_end()

    def __len__(self):
        return len(self.x1) // self.batch_size + 1

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, item):
        x1_batch = self.x1[self.indices[item * self.batch_size:(item + 1) * self.batch_size]]
        x2_batch = self.x2[self.indices[item * self.batch_size:(item + 1) * self.batch_size]]
        y_batch = self.y[self.indices[item * self.batch_size:(item + 1) * self.batch_size]]
        return [x1_batch, x2_batch], y_batch


def load_data(filename="./dataset/mitdb.pkl"):
    import pickle

    with open(filename, "rb") as f:
        (x1_train, x2_train, y_train), (x1_test, x2_test, y_test) = pickle.load(f)

    return (x1_train, x2_train, y_train), (x1_test, x2_test, y_test)


def main():
    epochs = 50
    batch_size = 512

    # loading data
    (x1_train, x2_train, y_train), (x1_test, x2_test, y_test) = load_data()

    x1_train = np.expand_dims(x1_train, axis=-1)
    x1_test = np.expand_dims(x1_test, axis=-1)

    scaler = RobustScaler()
    x2_train = scaler.fit_transform(x2_train)
    x2_test = scaler.transform(x2_test)

    y_train = keras.utils.to_categorical(y_train, num_classes=4)
    y_test = keras.utils.to_categorical(y_test, num_classes=4)

    print("train labels:", np.sum(y_train, axis=0))
    print("test labels:", np.sum(y_test, axis=0))

    train_generator = MyGenerator(x1_train, x2_train, y_train, batch_size)
    test_generator = MyGenerator(x1_test, x2_test, y_test, batch_size)

    model = create_model(l=1e-3)
    model.summary()

    # callbacks
    log_dir = os.path.join("./logs", datetime.now().strftime("%H-%M-%S"))
    tb_cb = TensorBoard(log_dir=log_dir)

    def schedule(epoch, lr):
        if (epoch + 1) % 10 == 0:
            lr *= 0.1
        return lr

    lr_scheduler = LearningRateScheduler(schedule=schedule, verbose=0)

    # training
    model.compile(loss=categorical_focal_loss(gamma=2), optimizer="adam", metrics=["acc", f1])
    model.fit_generator(train_generator, epochs=epochs, verbose=1, callbacks=[tb_cb, lr_scheduler],
                        validation_data=test_generator)

    model.save(os.path.join("./models", "model_focalloss.h5"))

    y_true = np.argmax(y_test, axis=-1)
    y_pred = np.argmax(model.predict([x1_test, x2_test], batch_size=batch_size, verbose=1), axis=-1)

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=3))


if __name__ == "__main__":
    main()
