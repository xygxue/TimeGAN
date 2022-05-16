"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

# Necessary Packages
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.layers import LayerNormalization, BatchNormalization
from keras.layers import GRU, Dense, Input, Dropout, LSTM
from keras.losses import BinaryCrossentropy, MeanSquaredError
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras import backend as K
from tqdm import tqdm

from metrics.visualization_metrics import loss_plot
import tensorflow_probability as tfp

tfd = tfp.distributions
from utils import grad_plot


def timegan (ori_data, parameters):
  """TimeGAN function.
  
  Use original data as training set to generater synthetic data (time-series)
  
  Args:
    - ori_data: original time-series data
    - parameters: TimeGAN network parameters
    
  Returns:
    - generated_data: generated time-series data
  """

  # Basic Parameters
  n_windows, seq_len, dim = np.asarray(ori_data).shape

  # Network Parameters
  hidden_dim = parameters['hidden_dim']
  num_layers = parameters['num_layer']
  iterations = parameters['iterations']
  batch_size = parameters['batch_size']
  scaler = parameters['scaler']
  dataset = parameters['data_name']
  acc_id = parameters['acc_id']
  gamma = 1
  intermediate_dim = 4

  # Device check
  gpu_devices = tf.config.experimental.list_physical_devices('GPU')
  if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
  else:
    print('Using CPU')

  # Create tf.data.Dataset
  real_series = (tf.data.Dataset
                 .from_tensor_slices(ori_data)
                 .shuffle(buffer_size=n_windows)
                 .batch(batch_size))
  real_series_iter = iter(real_series.repeat())

  # Set up random series generator
  def make_random_data():
    while True:
      yield np.random.uniform(low=0, high=1, size=(seq_len, dim))
  random_series = iter(tf.data.Dataset
                       .from_generator(make_random_data, output_types=tf.float32)
                       .batch(batch_size)
                       .repeat())
  # Set up Logger
  cur_date = parameters['cur_date']
  log_dir = os.path.join(Path(__file__).parent, 'log', f'{cur_date}')
  if not os.path.exists(log_dir):
    os.mkdir(log_dir)
  writer = tf.summary.create_file_writer(log_dir)

  ## Build a RNN networks
    
  # Input place holders
  X = Input(shape=[seq_len, dim], batch_size=batch_size, name='RealData')
  Z = Input(shape=[seq_len, dim], batch_size=batch_size, name='RandomData')

  # T = tf.placeholder(tf.int32, [None], name = "myinput_t")

  def make_rnn(n_layers, hidden_units, output_units, name):
    return Sequential(
                      [GRU(units=hidden_units,
                           return_sequences=True,
                           name=f'GRU_{i + 1}') for i in range(n_layers)] +
                      [Dense(units=output_units,
                             activation='sigmoid',
                             name='OUT')], name=name)

  def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """
    y_pred = tf.cast(y_pred, tf.float32)
    lh = tfp.distributions.Bernoulli(probs=y_pred)

    return -K.sum(lh.log_prob(y_true), axis=1)

  # Embedding network between original feature space to latent space.

  # embedder = make_rnn(n_layers=num_layers,
  #                     hidden_units=hidden_dim,
  #                     output_units=hidden_dim,
  #                     name='Embedder')
  embedder = Sequential([GRU(units=hidden_dim,
                             return_sequences=True,
                             name=f'GRU_{i + 1}') for i in range(num_layers)] +
                        [Dense(units=hidden_dim,
                               activation='sigmoid',
                               name='OUT')], name='Embedder')

  # Recovery network from latent space to original space.
  recovery_syn = make_rnn(n_layers=num_layers,
                      hidden_units=hidden_dim,
                      output_units=dim,
                      name='Recovery')
  recovery = Sequential([Dense(units=34,
                               activation='relu',
                               name='IN')] +
                        [GRU(units=hidden_dim,
                             return_sequences=True,
                             name=f'GRU_{i + 1}') for i in range(num_layers)] +
                        [Dense(units=dim,
                               activation='sigmoid',
                               name='OUT')], name='Recovery')
  # recovery = Sequential([GRU(units=hidden_dim,
  #                 return_sequences=True,
  #                 name=f'GRU_{i + 1}') for i in range(num_layers)] +
  #                        [Dense(units=dim,
  #                               activation='sigmoid',
  #                               name='OUT')] +
  #                        [tfp.layers.DistributionLambda(
  #                          lambda t: tfp.distributions.Normal(loc=t[..., 0],
  #                                               scale=0.01*tf.math.softplus(t[..., 1])))],
  #                       name='Recovery')

  # Generator function: Generate time-series data in latent space.
  # generator = make_rnn(n_layers=3,
  #                      hidden_units=hidden_dim,
  #                      output_units=hidden_dim,
  #                      name='Generator')
  generator = Sequential([GRU(units=hidden_dim,
                               return_sequences=True,
                               name=f'LSTM_{i + 1}') for i in range(12)] +
                         [Dropout(.2)] +
                         [LayerNormalization()] +
                         [LSTM(units=hidden_dim,
                               return_sequences=True,
                               name=f'GRU_{i + 1}') for i in range(num_layers+6)] +
                         [Dropout(.2)] +
                         [LayerNormalization()] +
                         [Dense(units=hidden_dim,
                                activation='tanh',
                                name='OUT')], name='Generator')
      
  # Generate next sequence using the previous sequence.

  supervisor = make_rnn(n_layers=2,
                        hidden_units=hidden_dim,
                        output_units=hidden_dim,
                        name='Supervisor')
          
  # Discriminate the original and synthetic time-series data.
  # discriminator = make_rnn(n_layers=num_layers-1,
  #                          hidden_units=hidden_dim,
  #                          output_units=1,
  #                          name='Discriminator')
  discriminator = Sequential([GRU(units=hidden_dim,
                                  return_sequences=True,
                                  name=f'GRU_{i + 1}') for i in range(num_layers-1)] +
                             [Dense(units=1,
                                    activation='sigmoid',
                                    name='OUT')], name='Discriminator')

  # Embedder & Recovery
  # H = embedder(X)
  # X_tilde = recovery(H)

  H = embedder(X)
  Z_mu = Dense(34, name='Dense_1')(H)
  Z_log_var = Dense(34, name='Dense_2')(H)

  Z_mu, Z_log_var = KLDivergenceLayer(name='KLDivergenceLayer')([Z_mu, Z_log_var])
  Z_sigma = Lambda(lambda t: K.exp(.5*t), name='Lambda')(Z_log_var)

  eps = Input(tensor=tf.random.normal(shape=(batch_size, seq_len, 34)), name='eps_input')
  Z_eps = Multiply(name='Multiply')([Z_sigma, eps])
  Z_encoder = Add(name='Add')([Z_mu, Z_eps])
  X_tilde = recovery(Z_encoder)

  autoencoder = Model(inputs=[X, eps],
                      outputs=X_tilde,
                      name='Autoencoder')
  # autoencoder.summary()
  # plot_model(autoencoder,
  #            to_file='model/autoencoder.png',
  #            show_shapes=True)


  # Generator
  # Adversarial Architecture - Supervised
  E_hat = generator(Z)
  H_hat = supervisor(E_hat)
  Y_fake = discriminator(H_hat)

  adversarial_supervised = Model(inputs=Z,
                                 outputs=Y_fake,
                                 name='AdversarialNetSupervised')
  # adversarial_supervised.summary()
  # plot_model(adversarial_supervised, to_file='model/adversarial_supervised.png', show_shapes=True)

  # Adversarial Architecture in Latent Space
  Y_fake_e = discriminator(E_hat)

  adversarial_emb = Model(inputs=Z,
                          outputs=Y_fake_e,
                          name='AdversarialNet')
  # adversarial_emb.summary()

  # plot_model(adversarial_emb, to_file='model/adversarial_emb.png', show_shapes=True)

  # Mean & Variance Loss
  # Synthetic data
  X_hat = recovery_syn(H_hat)
  synthetic_data = Model(inputs=Z,
                         outputs=X_hat,
                         name='SyntheticData')
  # synthetic_data.summary()
  # plot_model(synthetic_data, to_file='model/synthetic_data.png', show_shapes=True)

  # Discriminator
  # Architecture: Real Data
  Y_real = discriminator(H)
  discriminator_model = Model(inputs=X,
                              outputs=Y_real,
                              name='DiscriminatorReal')
  # discriminator_model.summary()
  # plot_model(discriminator_model, to_file='model/discriminator_model.png',show_shapes=True)

    
  # Generic Loss
  mse = MeanSquaredError()
  bce = BinaryCrossentropy()

  # Momment Loss
  def get_generator_moment_loss(y_true, y_pred):
    y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
    y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
    g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
    g_loss_var = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6)))
    return g_loss_mean + g_loss_var
    
  # optimizer
  autoencoder_optimizer = Adam()
  supervisor_optimizer = Adam()
  generator_optimizer = Adam()
  discriminator_optimizer = Adam()
  embedding_optimizer = Adam()

  ## TimeGAN training

  # 1. Embedding network training
  print('Start Embedding Network Training')

  @tf.function
  def train_autoencoder_init(x):
    with tf.GradientTape() as tape:
      eps_ = tf.random.normal(shape=(batch_size, seq_len, 34))
      x_tilde = autoencoder([x, eps_])

      embedding_loss_t0 = mse(x, x_tilde)
      e_loss_0 = 10 * tf.sqrt(embedding_loss_t0)

      # x_trans = tf.transpose(x, perm=[0, 2, 1])
      # x_tilde_trans = tf.transpose(x_tilde, perm=[0, 2, 1])
      # embedding_loss_t0 = mse(x[:, :, 0:1], x_tilde[:, :, 0:1])
      # embedding_loss_t1 = tf.math.reduce_mean(nll(x_trans[:, 1:3, :], x_tilde_trans[:, 1:3, :]), axis=1)
      # embedding_loss_t2 = tf.math.reduce_mean(nll(x_trans[:, 3:7, :], x_tilde_trans[:, 3:7, :]), axis=1)
      # embedding_loss_t3 = tf.math.reduce_mean(nll(x_trans[:, 7:11, :], x_tilde_trans[:, 7:11, :]), axis=1)
      # embedding_loss_t4 = tf.math.reduce_mean(nll(x_trans[:, 11:13, :], x_tilde_trans[:, 11:13, :]), axis=1)
      # embedding_loss_t5 = tf.math.reduce_mean(nll(x_trans[:, 13:, :], x_tilde_trans[:, 13:, :]), axis=1)
      # embedding_loss_ohe = embedding_loss_t1 + embedding_loss_t2 + embedding_loss_t3 + embedding_loss_t4 + embedding_loss_t5
      # e_loss_0 = 10 * tf.sqrt(embedding_loss_t0) + embedding_loss_ohe

    var_list = embedder.trainable_variables + recovery.trainable_variables
    gradients = tape.gradient(e_loss_0, var_list)
    autoencoder_optimizer.apply_gradients(zip(gradients, var_list))
    return tf.sqrt(embedding_loss_t0), gradients
    # return (tf.sqrt(embedding_loss_t0)+tf.math.reduce_mean(embedding_loss_ohe)), gradients


  autoencoder_grad = []
  for step in tqdm(range(int(iterations*1.2))):
    X_ = next(real_series_iter)
    if X_.shape[0] != batch_size:
      dim_diff = batch_size - X_.shape[0]
      X_trans_ = tf.transpose(X_, perm=[1, 0, 2])
      padded = tf.keras.layers.ZeroPadding1D(padding=(0, dim_diff))(X_trans_)
      X_ = tf.transpose(padded, perm=[1, 0, 2])
    step_e_loss_t0, emb_grad = train_autoencoder_init(X_)
    autoencoder_grad.append(emb_grad)
    with writer.as_default():
      tf.summary.scalar('Loss Autoencoder Init', step_e_loss_t0, step=step)
      
  print('Finish Embedding Network Training')

  # 2. Training only with supervised loss
  print('Start Training with Supervised Loss Only')

  # Train Step
  @tf.function
  def train_supervisor(x):
    with tf.GradientTape() as tape:
      h = embedder(x)
      h_hat_supervised = supervisor(h)
      g_loss_s = mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

    var_list = supervisor.trainable_variables + generator.trainable_variables
    gradients = tape.gradient(g_loss_s, var_list)
    apply_grads = [(grad, var) for (grad, var) in zip(gradients, var_list) if grad is not None]
    supervisor_optimizer.apply_gradients(apply_grads)
    return g_loss_s, gradients

  # Training Loop
  supervised_grad = []
  for step in tqdm(range(int(iterations*1.2))):
    X_ = next(real_series_iter)
    step_g_loss_s, sup_grad = train_supervisor(X_)
    supervised_grad.append(sup_grad)
    with writer.as_default():
      tf.summary.scalar('Loss Generator Supervised Init', step_g_loss_s, step=step)
  print('Finish Training with Supervised Loss Only')
    
  # 3. Joint Training
  print('Start Joint Training')

  # Generator Train Step
  @tf.function
  def train_generator(x, z):
    with tf.GradientTape() as tape:
      y_fake = adversarial_supervised(z)
      generator_loss_unsupervised = bce(y_true=tf.ones_like(y_fake),
                                        y_pred=y_fake)

      y_fake_e = adversarial_emb(z)
      generator_loss_unsupervised_e = bce(y_true=tf.ones_like(y_fake_e),
                                          y_pred=y_fake_e)
      h = embedder(x)
      h_hat_supervised = supervisor(h)
      generator_loss_supervised = mse(h[:, 1:, :], h_hat_supervised[:, -1:, :])

      x_hat = synthetic_data(z)
      generator_moment_loss = get_generator_moment_loss(x, x_hat)

      generator_loss = (generator_loss_unsupervised +
                        generator_loss_unsupervised_e +
                        100 * tf.sqrt(generator_loss_supervised) +
                        100 * generator_moment_loss)

    var_list = generator.trainable_variables + supervisor.trainable_variables
    gradients = tape.gradient(generator_loss, var_list)
    generator_optimizer.apply_gradients(zip(gradients, var_list))
    return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss, gradients

  # Embedding Train Step
  @tf.function
  def train_embedder(x):
    with tf.GradientTape() as tape:
      h = embedder(x)
      h_hat_supervised = supervisor(h)
      generator_loss_supervised = mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

      eps_ = tf.random.normal(shape=(batch_size, seq_len, 34))
      x_tilde = autoencoder([x, eps_])
      embedding_loss_t0 = mse(x, x_tilde)
      e_loss = 10 * tf.sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

      # x_trans = tf.transpose(x, perm=[0, 2, 1])
      # x_tilde_trans = tf.transpose(x_tilde, perm=[0, 2, 1])
      #
      # embedding_loss_t0 = mse(x[:, :, 0:1], x_tilde[:, :, 0:1])
      # embedding_loss_t1 = nll(x_trans[:, 1:3, :], x_tilde_trans[:, 1:3, :])
      # embedding_loss_t2 = nll(x_trans[:, 3:7, :], x_tilde_trans[:, 3:7, :])
      # embedding_loss_t3 = nll(x_trans[:, 7:11, :], x_tilde_trans[:, 7:11, :])
      # embedding_loss_t4 = nll(x_trans[:, 11:13, :], x_tilde_trans[:, 11:13, :])
      # embedding_loss_t5 = nll(x_trans[:, 13:, :], x_tilde_trans[:, 13:, :])
      # embedding_loss_ohe = embedding_loss_t1 + embedding_loss_t2 + embedding_loss_t3 + embedding_loss_t4 + embedding_loss_t5
      # e_loss = 10 * tf.sqrt(embedding_loss_t0) + embedding_loss_ohe + 0.1 * generator_loss_supervised

    var_list = embedder.trainable_variables + recovery.trainable_variables
    gradients = tape.gradient(e_loss, var_list)
    embedding_optimizer.apply_gradients(zip(gradients, var_list))
    # return (tf.sqrt(embedding_loss_t0)+tf.math.reduce_mean(embedding_loss_ohe)), gradients
    return tf.sqrt(embedding_loss_t0), gradients

  # Discriminator Train Step
  @tf.function
  def get_discriminator_loss(x, z):
    y_real = discriminator_model(x)
    discriminator_loss_real = bce(y_true=tf.ones_like(y_real),
                                  y_pred=y_real)

    y_fake = adversarial_supervised(z)
    discriminator_loss_fake = bce(y_true=tf.zeros_like(y_fake),
                                  y_pred=y_fake)

    y_fake_e = adversarial_emb(z)
    discriminator_loss_fake_e = bce(y_true=tf.zeros_like(y_fake_e),
                                    y_pred=y_fake_e)
    return (discriminator_loss_real +
            discriminator_loss_fake +
            gamma * discriminator_loss_fake_e)

  @tf.function
  def train_discriminator(x, z):
    with tf.GradientTape() as tape:
      discriminator_loss = get_discriminator_loss(x, z)

    var_list = discriminator.trainable_variables
    gradients = tape.gradient(discriminator_loss, var_list)
    discriminator_optimizer.apply_gradients(zip(gradients, var_list))
    return discriminator_loss, gradients

  # Training Loop
  d_loss = []
  g_loss_u = []
  g_loss_s = []
  g_loss_v = []
  e_loss_t0 = []
  step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0

  generator_grad = []
  joint_emb_grad = []
  discriminator_grad = []

  for step in tqdm(range(iterations)):
    # Train generator (twice as often as discriminator)
    for kk in range(2):
      X_ = next(real_series_iter)
      Z_ = next(random_series)
      if X_.shape[0] != batch_size:
        dim_diff = batch_size - X_.shape[0]
        X_trans_ = tf.transpose(X_, perm=[1, 0, 2])
        padded = tf.keras.layers.ZeroPadding1D(padding=(0, dim_diff))(X_trans_)
        X_ = tf.transpose(padded, perm=[1, 0, 2])
      # Train generator
      step_g_loss_u, step_g_loss_s, step_g_loss_v, gen_grad = train_generator(X_, Z_)
      generator_grad.append(gen_grad)

      # Train embedder
      step_e_loss_t0, emb_grad = train_embedder(X_)
      joint_emb_grad.append(emb_grad)

      g_loss_u.append(step_g_loss_u)
      g_loss_s.append(step_g_loss_s)
      g_loss_v.append(step_g_loss_v)
      e_loss_t0.append(step_e_loss_t0)

    X_ = next(real_series_iter)
    Z_ = next(random_series)
    step_d_loss = get_discriminator_loss(X_, Z_)
    if step_d_loss > 0.2:
      step_d_loss, disc_grad = train_discriminator(X_, Z_)
      discriminator_grad.append(disc_grad)
      d_loss.append(step_d_loss)

    if step % 1000 == 0:
      print(f'{step:6,.0f} | d_loss: {step_d_loss:6.4f} | g_loss_u: {step_g_loss_u:6.4f} | '
            f'g_loss_s: {step_g_loss_s:6.4f} | g_loss_v: {step_g_loss_v:6.4f} | e_loss_t0: {step_e_loss_t0:6.4f}')

    with writer.as_default():
      tf.summary.scalar('G Loss S', step_g_loss_s, step=step)
      tf.summary.scalar('G Loss U', step_g_loss_u, step=step)
      tf.summary.scalar('G Loss V', step_g_loss_v, step=step)
      tf.summary.scalar('E Loss T0', step_e_loss_t0, step=step)
      tf.summary.scalar('D Loss', step_d_loss, step=step)

  loss_plot(d_loss, g_loss_u, g_loss_s, g_loss_v, e_loss_t0, iterations, parameters['data_name'], cur_date)
  print('Finish Joint Training')

  # Persist Synthetic Data Generator
  synthetic_data.save(os.path.join(log_dir, 'synthetic_data'))

  ## Synthetic data generation
  generated_data = []
  for i in range(len(ori_data)):
    Z_ = next(random_series)
    d = synthetic_data(Z_)
    generated_data.append(d)

  generated_data = np.array(np.vstack(generated_data))
        
  # Renormalization
  generated_data = (scaler.inverse_transform(generated_data.reshape(-1, dim)).reshape(-1, seq_len, dim))

  return generated_data

class KLDivergenceLayer(Layer):

  """ Identity transform layer that adds KL divergence
  to the final model loss.
  """

  def __init__(self, *args, **kwargs):
    self.is_placeholder = True
    super(KLDivergenceLayer, self).__init__(*args, **kwargs)

  def call(self, inputs):

    mu, log_var = inputs

    kl_batch = - .5 * K.sum(1 + log_var -
                            K.square(mu) -
                            K.exp(log_var), axis=-1)

    self.add_loss(K.mean(kl_batch), inputs=inputs)

    return inputs