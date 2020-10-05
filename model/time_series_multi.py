import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def multivariate_data(dataset, target, start_index, end_index,
                      history_size, target_size, step=1, single_step=False):
    data = []
    labels = []

    N = 1
    if dataset.shape != target.shape:
        _, N = dataset.shape
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices].reshape(history_size, N))

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
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel('Time-Step')
    return plt


def create_dataset(df):
    tf.random.set_seed(13)

    # get temperature data
    features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
    features = df[features_considered]
    features.index = df['Date Time']
    dataset = features.values
    return dataset


def normalize_data(data):
    # normalize dataset
    data_mean = data[:TRAIN_SPLIT].mean(axis=0)
    data_std = data[:TRAIN_SPLIT].std(axis=0)
    data = (data - data_mean) / data_std
    return data


def create_model(x, dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=x.shape[-2:]),
        tf.keras.layers.LSTM(32, return_sequences=True, ),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(dim)
    ])
    return model


def save_training_history(history, model, epochs, past, future, name='mape_multi_step_3LSTM_64_'):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.clf()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    img_name = name + str(past) + '_' + str(future) + '.jpg'
    plt.savefig(img_name)

    hist_name = name + str(past) + '_' + str(future) + '.npy'
    np.save(hist_name, history.history)

    model_name = name + str(past) + '_' + str(future) + '.hdf5'
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

    TRAIN_TEST_SPLIT = int(np.ceil(0.8*len(df)))
    TRAIN_SPLIT = TRAIN_TEST_SPLIT - int(np.ceil(0.25*TRAIN_TEST_SPLIT))

    dataset = create_dataset(df)
    dataset = normalize_data(dataset)

    # create train and valid
    past_history = 100
    future_target = 30
    STEP = 1

    x_train_single, y_train_single = multivariate_data(dataset[:, 1], dataset[:, 1], 0,
                                                       TRAIN_SPLIT, past_history,
                                                       future_target, STEP,
                                                       single_step=False)
    x_val_single, y_val_single = multivariate_data(dataset[:, 1], dataset[:, 1],
                                                   TRAIN_SPLIT, TRAIN_TEST_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=False)
    x_test_single, y_test_single = multivariate_data(dataset[:, 1], dataset[:, 1],
                                                   TRAIN_TEST_SPLIT, None, past_history,
                                                   future_target, STEP,
                                                   single_step=False)

    print('\n Train shape {} '.format(x_train_single.shape))
    print('Valid shape {} '.format(x_val_single.shape))
    print('Test shape {} '.format(x_test_single.shape))


    print('\n Single window of past history : {}'.format(x_train_single[0].shape))
    print ('Single window of future target : {}'.format(y_train_single[0].shape))

    BATCH_SIZE = 256
    BUFFER_SIZE = 10000

    train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
    train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

    mymodel = create_model(x_train_single, future_target)
    mymodel.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mean_absolute_percentage_error', 
                    metrics=['mae', 'mse'])

    # take the first 256 data (batch size = 256)
    for x, y in val_data_single.take(1):
        print('\n First training input x : {} and y : {}:'.format(x.shape, y.shape))

    EVALUATION_INTERVAL = int(len(x_train_single)//BATCH_SIZE)
    EPOCHS = 25

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5)

    # history = mymodel.fit(train_data_single, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL,
    #                       callbacks=[early_stopping], validation_data=val_data_single, 
    #                       validation_steps=int(len(x_val_single)//BATCH_SIZE))
    history = mymodel.fit(x_train_single, y_train_single, epochs=EPOCHS, callbacks=[early_stopping], 
                          validation_data=(x_val_single, y_val_single), batch_size=BATCH_SIZE)

    # plot loss over epochs
    save_training_history(history, mymodel, EPOCHS,
                          past_history, future_target)

    test_loss = mymodel.evaluate(x_test_single, y_test_single, batch_size=BATCH_SIZE)
    # test_loses_10_3lstm.append(test_loss)
    print('test loss :', test_loss)

    # features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
