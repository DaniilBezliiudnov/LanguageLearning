from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from keras.layers.preprocessing import image_preprocessing as p_layers
from matplotlib import pyplot as plt
import numpy as np
from fuzzywuzzy import fuzz


def print_history(history):
    plt.plot(history.epoch, history.history['accuracy'])
    plt.plot(history.epoch, history.history['loss'])
    plt.plot(history.epoch, history.history['val_accuracy'])
    plt.plot(history.epoch, history.history['val_loss'])
    plt.legend(["accuracy", "loss", "val_accuracy", "val_loss"])
    plt.title("accuracy/loss function")
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("accuracy/loss")
    plt.show()


def fuzzy_checker(seq1, seq2):
    chars = "".join(list(map(chr, seq1)))
    chars2 = "".join(list(map(chr, seq2)))
    return fuzz.ratio(chars, chars2)


def create_name_branch(i_name_1, i_name_2):
    i_name_1_rs = p_layers.Rescaling(scale=1./61, offset=-1)(i_name_1)
    l1_name1 = layers.LocallyConnected1D(10, 3, activation='relu')(i_name_1_rs)
    i_name_2_rs = p_layers.Rescaling(scale=1./61, offset=-1)(i_name_2)
    l1_name2 = layers.LocallyConnected1D(10, 3, activation='relu')(i_name_2_rs)
    l2_names = layers.Concatenate()([l1_name1, l1_name2])
    l3_names = layers.Flatten()(l2_names)
    return layers.Dense(5, activation='tanh')(l3_names)


def create_dob_branch(i_dob_1, i_dob_2):
    i_dob_1_rs = p_layers.Rescaling(scale=1./29, offset=-1)(i_dob_1)
    l1_dob1 = layers.LocallyConnected1D(5, 3, activation='relu')(i_dob_1_rs)
    i_dob_2_rs = p_layers.Rescaling(scale=1./29, offset=-1)(i_dob_2)
    l1_dob2 = layers.LocallyConnected1D(5, 3, activation='relu')(i_dob_2_rs)
    l2_dobs = layers.Concatenate()([l1_dob1, l1_dob2])
    l2_dobs = layers.Flatten()(l2_dobs)
    return layers.Dense(10, activation='tanh')(l2_dobs)


def create_gender_branch(i_gender_1, i_gender_2):
    i_gender_1_rs = p_layers.Rescaling(scale=1./61, offset=-1)(i_gender_1)
    i_gender_2_rs = p_layers.Rescaling(scale=1./61, offset=-1)(i_gender_2)
    l2_genders = layers.Concatenate()([i_gender_1_rs, i_gender_2_rs])
    return layers.Dense(10, activation='tanh')(l2_genders)


def create_ratio_branch(i_ratio, dense_units):
    i_ratio_n_rs = p_layers.Rescaling(scale=1./50, offset=-1)(i_ratio)
    return layers.Dense(dense_units, activation='tanh')(i_ratio_n_rs)


def create_model_v2(data):
    name_len = len(data['training_data'][0][0])
    gender_len = len(data['training_data'][0][1])
    dob_len = len(data['training_data'][0][2])

    i_name_1 = keras.Input(name="name_1", shape=(name_len, 1))
    i_name_2 = keras.Input(name="name_2", shape=(name_len, 1))
    i_gender_1 = keras.Input(name="gender_1", shape=(gender_len, ))
    i_gender_2 = keras.Input(name="gender_2", shape=(gender_len, ))
    i_dob_1 = keras.Input(name="dob_1", shape=(dob_len, 1))
    i_dob_2 = keras.Input(name="dob_2", shape=(dob_len, 1))
    i_ratio_n = keras.Input(name="fuzz_n", shape=(1, ))
    i_ratio_g = keras.Input(name="fuzz_g", shape=(1, ))

    l4_combined = layers.Concatenate()([
        create_name_branch(i_name_1, i_name_2),
        create_gender_branch(i_gender_1, i_gender_2),
        create_dob_branch(i_dob_1, i_dob_2),
        create_ratio_branch(i_ratio_n, 10),
        create_ratio_branch(i_ratio_g, 5)])
    l5_brain = layers.Dense(20, activation='relu')(l4_combined)

    model = keras.Model(inputs=[
        i_name_1,
        i_gender_1,
        i_dob_1,
        i_name_2,
        i_gender_2,
        i_dob_2,
        i_ratio_n,
        i_ratio_g
    ], outputs=[
        layers.Dense(1, activation='sigmoid')(l5_brain)
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


def to_dict(seq):
    return {
        "name_1": np.asarray(list(map(lambda x: np.asarray(x[0]), seq))),
        "gender_1":  np.asarray(list(map(lambda x: np.asarray(x[1]), seq))),
        "dob_1":  np.asarray(list(map(lambda x: np.asarray(x[2]), seq))),
        "name_2":  np.asarray(list(map(lambda x: np.asarray(x[3]), seq))),
        "gender_2":  np.asarray(list(map(lambda x: np.asarray(x[4]), seq))),
        "dob_2":  np.asarray(list(map(lambda x: np.asarray(x[5]), seq))),
        "fuzz_n": np.asarray(list(map(lambda x: fuzzy_checker(x[0], x[3]), seq))),
        "fuzz_g": np.asarray(list(map(lambda x: fuzzy_checker(x[2], x[5]), seq)))
    }


def train_model(model: keras.Sequential, data, epochs):
    logdir = f'logs/fit/{epochs}/' + \
        datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
    early_stop_callback = callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10, min_delta=0.0001, restore_best_weights=True, verbose=0)
    learning_rate_callback = callbacks.LearningRateScheduler(
        lambda epoch, lr: lr if epoch < 16 else lr * 0.93, verbose=0)

    x_train = to_dict(data['training_data'])
    y_train = tf.convert_to_tensor(data['training_labels'])
    x_val = to_dict(data['test_data'])
    y_val = tf.convert_to_tensor(data['test_labels'])
    training_history = model.fit(
        x_train,
        y_train,
        # batch_size=30,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[tensorboard_callback,
                   early_stop_callback, learning_rate_callback]
    )

    return training_history
