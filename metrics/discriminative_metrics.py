"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use post-hoc RNN to classify original data and synthetic data

Output: discriminative score (np.abs(classification accuracy - 0.5))
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.metrics import AUC
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense


def ts_classification_training_plot(history, dataset, cur_date):
    plot_path = os.path.join(Path(__file__).parents[1], 'results', dataset, 'discriminative_metrics', '{cur_date}.png')


    fig, axes = plt.subplots(ncols=2, figsize=(14, 4))
    history[['AUC', 'val_AUC']].rename(columns={'AUC': 'Train', 'val_AUC': 'Test'}).plot(ax=axes[1],
                                                                                         title='ROC Area under the Curve',
                                                                                         style=['-', '--'],
                                                                                         xlim=(0, 250))
    history[['accuracy', 'val_accuracy']].rename(columns={'accuracy': 'Train', 'val_accuracy': 'Test'}).plot(ax=axes[0],
                                                                                                             title='Accuracy',
                                                                                                             style=['-',
                                                                                                                    '--'],
                                                                                                             xlim=(
                                                                                                             0, 250))
    for i in [0, 1]:
        axes[i].set_xlabel('Epoch')

    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    axes[0].set_ylabel('Accuracy (%)')
    axes[1].set_ylabel('AUC')
    fig.suptitle('Assessing Fidelity: Time Series Classification Performance', fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=.85)
    plt.savefig(plot_path.format(cur_date=cur_date))


def discriminative_score_metrics(ori_data, generated_data, dataname, cur_date):
    """Use post-hoc RNN to classify original data and synthetic data
  
    Args:
      - ori_data: original data
      - generated_data: generated synthetic data

    Returns:
      - discriminative_score: np.abs(classification accuracy - 0.5)
    """
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    # Basic Parameters
    n_series, seq_len, dim = ori_data.shape

    # Train test split
    idx = np.arange(n_series)
    n_train = int(.8 * n_series)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    train_data = np.vstack((ori_data[train_idx],
                            generated_data[train_idx]))
    test_data = np.vstack((ori_data[test_idx],
                           generated_data[test_idx]))
    n_train, n_test = len(train_idx), len(test_idx)

    # Classification labels
    train_labels = np.concatenate((np.ones(n_train),
                                   np.zeros(n_train)))
    test_labels = np.concatenate((np.ones(n_test),
                                  np.zeros(n_test)))

    # Build a post-hoc RNN discriminator network
    # discriminator/classifier function
    ts_classifier = Sequential([GRU(36, input_shape=(24, dim), name='GRU'),
                                Dense(1, activation='sigmoid', name='OUT')],
                               name='Time_Series_Classifier')
    ts_classifier.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=[AUC(name='AUC'), 'accuracy'])
    result = ts_classifier.fit(x=train_data,
                               y=train_labels,
                               validation_data=(test_data, test_labels),
                               epochs=250,
                               batch_size=36,
                               verbose=0)
    history = pd.DataFrame(result.history)
    discriminative_score = ts_classifier.evaluate(x=test_data, y=test_labels)
    ts_classification_training_plot(history, dataname, cur_date)
    return discriminative_score
