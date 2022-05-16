"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use Post-hoc RNN to predict one-step ahead (last feature)
"""
import os
from datetime import datetime
from pathlib import Path

import numpy as np
# Necessary Packages
import pandas as pd
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.losses import MeanAbsoluteError


def ts_pred_plot(synthetic_result, real_result, dataset, cur_date):
    plot_path = os.path.join(Path(__file__).parents[1], 'results', dataset, 'predictive_metrics', '{cur_date}.png')
    fig, axes = plt.subplots(ncols=2, figsize=(14, 4))
    synthetic_result.plot(ax=axes[0], title='Train on Synthetic, Test on Real', logy=True, xlim=(0, 100))
    real_result.plot(ax=axes[1], title='Train on Real, Test on Real', logy=True, xlim=(0, 100))
    for i in [0, 1]:
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Mean Absolute Error (log scale)')

    fig.suptitle('Assessing Usefulness: Time Series Prediction Performance', fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=.85)
    plt.savefig(plot_path.format(cur_date=cur_date))


def predictive_score_metrics(ori_data, generated_data, dataname, cur_date):
    """Report the performance of Post-hoc RNN one-step ahead prediction.

    Args:
    - ori_data: original data
    - generated_data: generated synthetic data

    Returns:
    - predictive_score: MAE of the predictions on the original data
    """
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    # Basic Parameters
    n_series, seq_len, dim = np.asarray(ori_data).shape
    idx = np.arange(n_series)
    n_train = int(.8 * n_series)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    real_train_data = ori_data[train_idx, :, :dim-1]
    real_train_label = ori_data[train_idx, :, -1]

    real_test_data = ori_data[test_idx, :, :dim-1]
    real_test_label = ori_data[test_idx, :, -1]

    synthetic_train = generated_data[:, :, :dim-1]
    synthetic_label = generated_data[:, :, -1]

    def get_model():
      model = Sequential([GRU(12, input_shape=(seq_len, dim-1)),
                          Dense(24)])

      model.compile(optimizer=Adam(),
                    loss=MeanAbsoluteError(name='MAE'))
      return model


    ts_regression = get_model()
    synthetic_result = ts_regression.fit(x=synthetic_train,
                                         y=synthetic_label,
                                         validation_data=(
                                           real_test_data,
                                           real_test_label),
                                         epochs=100,
                                         batch_size=36,
                                         verbose=0)
    ts_regression = get_model()
    real_result = ts_regression.fit(x=real_train_data,
                                    y=real_train_label,
                                    validation_data=(
                                      real_test_data,
                                      real_test_label),
                                    epochs=100,
                                    batch_size=36,
                                    verbose=0)
    synthetic_result = pd.DataFrame(synthetic_result.history).rename(columns={'loss': 'Train', 'val_loss': 'Test'})
    real_result = pd.DataFrame(real_result.history).rename(columns={'loss': 'Train', 'val_loss': 'Test'})
    ts_pred_plot(synthetic_result, real_result, dataname, cur_date)
    return synthetic_result, real_result
