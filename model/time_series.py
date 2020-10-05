import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def univariate_data(dataset, start_index, end_index, history_size, 
                    target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
      end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
      indices = range(i-history_size, i, step)
      # Reshape data from (history_size,) to (history_size, 1)
      data.append(dataset[indices])
      if single_step:
        labels.append(dataset[i+target_size])
      else:
        labels.append(dataset[i:i+target_size])
  return np.array(data), np.array(labels)

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

def create_time_steps(length):
  return list(range(-length, 0))

def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

def create_univariate_dataset(df):
  tf.random.set_seed(13)

  # get temperature data
  uni_data = df['T (degC)']
  uni_data.index = df['Date Time']
  uni_data = uni_data.values
  return uni_data

def normalize_data(data):
  # normalize dataset
  data_mean = data[:TRAIN_SPLIT].mean()
  data_std = data[:TRAIN_SPLIT].std()
  data = (data-data_mean)/data_std
  return data

def create_model(x):
  model = tf.keras.models.Sequential([
          tf.keras.layers.LSTM(32, input_shape=x.shape[-2:]),
          tf.keras.layers.Dense(1)
          ])
  return model

def save_training_history(history, model, epochs, past, future, n=1):
  history_dict = history.history
  loss_values = history_dict['loss']
  val_loss_values = history_dict['val_loss']

  epochs = range(1, epochs+1)

  plt.clf()
  plt.plot(epochs, loss_values, 'o', label='Training loss')
  plt.plot(epochs, val_loss_values, label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  img_name = str(n)+'_'+str(past)+'_'+str(future)+'_3lstm'+'.jpg'
  plt.savefig(img_name)

  hist_name = str(n)+'_'+str(past)+'_'+str(future)+'_3lstm'+'.npy'
  np.save(hist_name, history.history)

  model_name = str(n)+'_'+str(past)+'_'+str(future)+'_3lstm'+'.hdf5'
  tf.keras.models.save_model(model=model, filepath=model_name)

  print('Saving plot, loss history, and model finished...')

if __name__ == "__main__":
  zip_path = tf.keras.utils.get_file(
      origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
      fname='jena_climate_2009_2016.csv.zip',
      extract=True)
  csv_path, _ = os.path.splitext(zip_path)

  df = pd.read_csv(csv_path)
  print('finished reading {} records'.format(len(df)))

  TRAIN_SPLIT = 300000
  uni_data = create_univariate_dataset(df)
  uni_data = normalize_data(uni_data)

  # create train and valid
  univariate_past_history = 720
  univariate_future_target = 72
  step = 6

  x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                            univariate_past_history,
                                            univariate_future_target, step)

  x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                        univariate_past_history,
                                        univariate_future_target, step)

  print ('Single window of past history')
  print (x_train_uni[0])
  print ('\n Target temperature to predict')
  print (y_train_uni[0])
  print ('\n Train size : {}, Valid size : {}'.format((x_train_uni.shape, y_train_uni.shape), (x_val_uni.shape, y_val_uni.shape)))

  # show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')

  BATCH_SIZE = 256
  BUFFER_SIZE = 10000

  train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
  train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

  val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
  val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

  simple_lstm_model = create_model(x_train_uni)
  simple_lstm_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

  # take the first 256 data (batch size = 256)
  for x, y in val_univariate.take(1):
    print('\n First training input {}:'.format(x.shape, y.shape))
    print(simple_lstm_model.predict(x).shape)

  EVALUATION_INTERVAL = 200
  EPOCHS = 10

  history = simple_lstm_model.fit(train_univariate, epochs=EPOCHS, 
                              steps_per_epoch=EVALUATION_INTERVAL,
                              validation_data=val_univariate, validation_steps=50)
  # plot loss over epochs
  save_training_history(history, simple_lstm_model, EPOCHS, 
                        univariate_past_history, univariate_future_target)

  # test_loss = simple_lstm_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
  # test_loses_10_3lstm.append(test_loss)
  # print('test loss :', test_loss)
  
  # features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']