import time
import tracemalloc

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from losses import MeanSquaredError
from lm import LMWrapper
from random import shuffle


def train_and_evaluate(model: tf.keras.Model, x, y, num_epochs: int = 1000, batch_size: int = 32):
    hist = model.fit(x, y, epochs=num_epochs, batch_size=batch_size)
    return hist.history


def main():
    num_experiments = 1
    n = 1000
    X = np.linspace(-5, 5, n).reshape(-1, 1)
    # Y = 4 * X * np.sin(10 * X) + X * np.exp(X) * np.cos(20 * X)
    Y = 0.05 + (0.05 * X ** 2 - 0.05 * np.cos(2 * np.pi * X))

    idxs = list(range(n))
    shuffle(idxs)
    x = X[idxs]
    y = Y[idxs]

    for experiment in range(num_experiments):
        adam_results = []
        lm_results = []

        # tracemalloc.start()
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(units=10, activation="tanh"),
            tf.keras.layers.Dense(units=1, activation="linear"),
        ])

        model.summary()

        # model_lm = LMWrapper(tf.keras.models.clone_model(model))
        model_lm = LMWrapper(model)
        model_lm.set_weights(model.get_weights())

        # Training Adam network
        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        #               loss=tf.keras.losses.MeanSquaredError(),
        #               metrics=[tf.keras.metrics.RootMeanSquaredError()])

        # t1_start = time.perf_counter()
        # adam_hist = train_and_evaluate(model, x, y, batch_size=1000)
        # t1_stop = time.perf_counter()

        # print(f"Adam: {(t1_stop - t1_start)}")

        # We use GD to discount the update factor of the LM algorithm, if we set the learning rate to
        # 1.0, it would just use the LM updated weights without discounting.
        model_lm.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=1.),
            loss=MeanSquaredError(),
            experimental_use_pfor=False,
            metrics=[tf.keras.metrics.RootMeanSquaredError()])

        t2_start = time.perf_counter()
        lm_hist = train_and_evaluate(model_lm, x, y, num_epochs=2000, batch_size=n)
        t2_stop = time.perf_counter()

        print(f"Levenberg-Marquardt per layer: {(t2_stop - t2_start)}")

        # adam_hist["time"] = t1_stop - t1_start
        # lm_hist["time"] = t2_stop - t2_start

        # adam_results.append({
        #     "time": t1_stop - t1_start,
        #     "loss": adam_hist["loss"],
        #     "rmse": adam_hist["root_mean_squared_error"]
        # })

        # lm_results.append({
        #     "time": t2_stop - t2_start,
        #     "loss": lm_hist["loss"],
        #     "attempts": lm_hist["attempts"],
        #     "rmse": lm_hist["root_mean_squared_error"]
        # })

        # adam_df = pd.DataFrame.from_records(adam_results)
        # lm_df = pd.DataFrame.from_records(lm_results)

        # adam_df.to_csv("./results/adam_results.csv", index=False)
        # lm_df.to_csv("./results/lm_results.csv", index=False)

        # print(tracemalloc.get_traced_memory())
        # tracemalloc.stop()
        y_hat = model_lm.predict(x)

        plt.scatter(x, y_hat, label="prediction", s=0.3)
        plt.plot(X, Y, label="true")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
