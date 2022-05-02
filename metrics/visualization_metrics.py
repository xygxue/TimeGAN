"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

visualization_metrics.py

Note: Use PCA or tSNE for generated and original data visualization
"""

# Necessary packages
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def visualization (ori_data, generated_data, analysis, dataset):
  """Using PCA or tSNE for generated and original data visualization.

  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
  """

  plot_path = os.path.join(Path(__file__).parents[1], 'results', dataset, 'plot', '{analysis}_{cur_date}.png')


  # Analysis sample size (for faster computation)
  anal_sample_no = min([1000, len(ori_data)])
  idx = np.random.permutation(len(ori_data))[:anal_sample_no]

  # Data preprocessing
  ori_data = np.asarray(ori_data)
  generated_data = np.asarray(generated_data)

  ori_data = ori_data[idx]
  generated_data = generated_data[idx]

  no, seq_len, dim = ori_data.shape

  for i in range(anal_sample_no):
    if (i == 0):
      prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
      prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
    else:
      prep_data = np.concatenate((prep_data,
                                  np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
      prep_data_hat = np.concatenate((prep_data_hat,
                                      np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))

  # Visualization parameter
  colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

  now = datetime.now()
  cur_date = now.strftime("%Y-%m-%d-%H")

  if analysis == 'pca':
    # PCA Analysis
    pca = PCA(n_components = 2)
    pca.fit(prep_data)
    pca_results = pca.transform(prep_data)
    pca_hat_results = pca.transform(prep_data_hat)

    # Plotting
    f, ax = plt.subplots(1)
    plt.scatter(pca_results[:,0], pca_results[:,1],
                c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
    plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1],
                c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")

    ax.legend()
    plt.title('PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    plt.savefig(plot_path.format(analysis=analysis, cur_date=cur_date))
    plt.show()

  elif analysis == 'tsne':

    # Do t-SNE Analysis together
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)

    # TSNE anlaysis
    tsne = TSNE(n_components=2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(prep_data_final)

    # Plotting
    f, ax = plt.subplots(1)

    plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1],
                c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
    plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1],
                c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")

    ax.legend()

    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.savefig(plot_path.format(analysis=analysis, cur_date=cur_date))
    plt.show()


def loss_plot(d_loss, g_loss_u, g_loss_s, g_loss_v, e_loss_t0, iterations, dataset, cur_date):

    plot_path = os.path.join(Path(__file__).parents[1], 'results', dataset, 'loss', '{cur_date}.png')

    plt.plot(range(len(d_loss)), d_loss, label="Disriminator Loss")
    plt.plot(range(len(g_loss_u)), g_loss_u, label="Generator Adversarial Loss")
    plt.plot(range(len(g_loss_u)), g_loss_s, label="Generator Supervised(Hidden) Loss")
    plt.plot(range(len(g_loss_u)), g_loss_v, label="Generator Moment Loss")
    plt.plot(range(len(g_loss_u)), e_loss_t0, label="Embedder Loss")
    plt.legend()
    plt.savefig(plot_path.format(cur_date=cur_date))
    plt.show()


def set_style(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def compare_hists(x_real, x_fake, ax=None, label=None):
    """ Computes histograms and plots those. """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if label is not None:
        label_historical = 'Historical ' + label
        label_generated = 'Generated ' + label
    else:
        label_historical = 'Historical'
        label_generated = 'Generated'
    bin_edges = ax.hist(x_real.flatten(), bins=80, alpha=0.6, density=True, label=label_historical)[1]
    ax.hist(x_fake.flatten(), bins=bin_edges, alpha=0.6, density=True, label=label_generated)
    ax.grid()
    set_style(ax)
    ax.legend()
    ax.set_ylabel('pdf')
    return ax


def fake_real_dist_plot(x_real, x_fake, plot_path, fake_filename, labels):
    # plot the histogram for each column in fake and real data
    directory = os.path.splitext(fake_filename)[0]
    path = os.path.join(plot_path, directory)
    if not os.path.exists(path):
        os.mkdir(path)
    x_real = np.asarray(x_real)
    for c in range(x_fake.shape[2]):
        fake_col = x_fake[:, :, c]
        real_col = x_real[:, :, c]
        ax = compare_hists(real_col, fake_col, ax=None, label=labels[c])
        ax.plot()
        plt.savefig(os.path.join(plot_path, directory, f"{labels[c]}.png"))


def ts_real_fake_plot(x_real, x_fake,  plot_path, fake_filename, labels, cur_date):
    directory = os.path.splitext(fake_filename)[0]
    path = os.path.join(plot_path, directory)
    if not os.path.exists(path):
        os.mkdir(path)
    nrows = len(labels)//2 + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(15, 10))
    axes = axes.flatten()

    obs = np.random.randint(len(x_real))

    for j, col in enumerate(labels):
        df = pd.DataFrame({'Real': x_real[obs][:, j],
                           'Synthetic': x_fake[obs][:, j]})
        df.plot(ax=axes[j],
                title=col,
                secondary_y='Synthetic data', style=['-', '--'])
    fig.savefig(os.path.join(plot_path, directory, f"{cur_date}.png"))
    fig.tight_layout()
