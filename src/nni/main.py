import sys
import tensorflow as tf
import tensorflow.keras as k
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from nni import get_next_parameter, report_intermediate_result, report_final_result
import tempfile


class ReportIntermediates(k.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        report_intermediate_result({'val_loss': logs['val_loss'], 'loss': logs['loss']})


def make_dataset(feature, y, investment_id, batch_size, mode="train"):
    ds = tf.data.Dataset.from_tensor_slices(((investment_id, feature), y))
    # ds = tf.data.Dataset.from_tensor_slices((feature, y))
    if mode == "train":
        ds = ds.shuffle(4096)
    ds = ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_model(investment_id, activation, lr, dropout, dropout2):
    investment_ids = list(investment_id.unique())
    investment_id_size = len(investment_ids) + 1
    investment_id_lookup_layer = k.layers.IntegerLookup(max_tokens=investment_id_size)
    investment_id_lookup_layer.adapt(pd.DataFrame({"investment_ids": investment_ids}))

    investment_id_inputs = k.Input((1, ), dtype=tf.uint16)
    investment_id_x = investment_id_lookup_layer(investment_id_inputs)
    investment_id_x = k.layers.Embedding(investment_id_size, 32, input_length=1)(investment_id_x)
    investment_id_x = k.layers.Reshape((-1, ))(investment_id_x)
    investment_id_x = k.layers.Dense(64, activation=activation)(investment_id_x)
    investment_id_x = k.layers.Dropout(dropout2)(investment_id_x)
    investment_id_x = k.layers.Dense(64, activation=activation)(investment_id_x)
    investment_id_x = k.layers.Dropout(dropout2)(investment_id_x)
    investment_id_x = k.layers.Dense(64, activation=activation)(investment_id_x)
    investment_id_x = k.layers.Dropout(dropout2)(investment_id_x)

    feature_inputs = k.Input((300,), dtype=tf.float16)
    feature_x = k.layers.Dense(256, activation=activation)(feature_inputs)
    feature_x = k.layers.Dropout(dropout2)(feature_x)
    feature_x = k.layers.Dense(256, activation=activation)(feature_x)
    feature_x = k.layers.Dropout(dropout2)(feature_x)
    feature_x = k.layers.Dense(256, activation=activation)(feature_x)
    feature_x = k.layers.Dropout(dropout2)(feature_x)

    x = k.layers.Concatenate(axis=1)([investment_id_x, feature_x])
    x = k.layers.Dense(512, activation=activation)(x)
    x = k.layers.Dropout(dropout)(x)
    x = k.layers.Dense(128, activation=activation)(x)
    feature_x = k.layers.Dropout(dropout2)(feature_x)
    x = k.layers.Dense(32, activation=activation)(x)
    feature_x = k.layers.Dropout(dropout2)(feature_x)
    output = k.layers.Dense(1)(x)

    model = k.Model(inputs=[investment_id_inputs, feature_inputs], outputs=[output])
    model.compile(optimizer=k.optimizers.Adam(lr), loss='mse', metrics=['mse', 'mae'])
    return model


def fit(model, train, features, investment_id, y, epochs, batch_size):
    kfold = KFold(5, shuffle=True, random_state=23)
    for i, (train_indices, target_indices) in enumerate(kfold.split(train[features], investment_id)):
        X_train, X_val = train.loc[train_indices, features], train.loc[target_indices, features]
        y_train, y_val = y.loc[train_indices], y.loc[target_indices]
        investment_id_train, investment_id_val = investment_id.loc[train_indices], investment_id.loc[target_indices]

        train_ds = make_dataset(X_train, y_train, investment_id_train, batch_size=batch_size)
        val_ds = make_dataset(X_val, y_val, investment_id_val, batch_size=batch_size, mode='val')

        if True:
            chkdir = tempfile.mkdtemp()
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=[ReportIntermediates(),
                           k.callbacks.EarlyStopping(patience=10),
                           k.callbacks.ModelCheckpoint(chkdir, save_best_only=True)]
            )

        model = k.models.load_model(chkdir)
        pearson_score = stats.pearsonr(model.predict(val_ds).ravel(), y_val.values)
        report_final_result({"default": pearson_score[0], "pearson score": pearson_score[0], "loss": np.min(history.history["val_loss"])})

        # pd.DataFrame(history.history, columns=['mse', 'val_mse']).plot()
        # plt.title("MSE")
        # pd.DataFrame(history.history, columns=['mae', 'val_mae']).plot()
        # plt.title("MAE")
        # plt.show()
        break


if __name__ == '__main__':
    nni_params = get_next_parameter()
    lr = nni_params['lr']
    batch_size = nni_params['bs']
    dropout = nni_params['dropout']
    dropout2 = nni_params['dropout2']

    epochs = 40
    activation = 'relu'
    kernel_regularizer = None
    features = [f'f_{i}' for i in range(300)]
    # activation = 'swish'
    # kernel_regularizer = 'l2'

    train = pd.read_pickle('/home/yanhuize/test/inputs/dataset/train.pkl')
    investment_id = train.pop('investment_id')
    time_id = train.pop('time_id')
    y = train.pop('target')

    model = get_model(investment_id, activation, lr, dropout, dropout2)
    fit(model, train, features, investment_id, y, epochs, batch_size)
