from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from matplotlib import pyplot as plt
from pipe import select
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


def create_model(data):
    model = keras.Sequential([
        keras.Input(shape=(len(data['training_data'][0]), )),
        layers.Dense(30, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # adam = keras.optimizers.Adam(learning_rate=0.0001,
    #                             beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def create_model_v1p5(data):
    name_len = len(data['training_data'][0][0])
    gender_len = len(data['training_data'][0][1])
    dob_len = len(data['training_data'][0][2])

    input_name_1 = keras.Input(name="name_1", shape=(name_len, ))
    input_name_2 = keras.Input(name="name_2", shape=(name_len, ))
    input_gender_1 = keras.Input(name="gender_1", shape=(gender_len, ))
    input_gender_2 = keras.Input(name="gender_2", shape=(gender_len, ))
    input_dob_1 = keras.Input(name="dob_1", shape=(dob_len, ))
    input_dob_2 = keras.Input(name="dob_2", shape=(dob_len, ))

    layer_1_input_name_1 = layers.Dense(
        name_len, activation='relu')(input_name_1)
    layer_1_input_name_2 = layers.Dense(
        name_len, activation='relu')(input_name_2)
    layer_1_input_dob_1 = layers.Dense(dob_len, activation='relu')(input_dob_1)
    layer_1_input_dob_2 = layers.Dense(dob_len, activation='relu')(input_dob_2)

    layer_2_names = layers.Concatenate()(
        [layer_1_input_name_1, layer_1_input_name_2])
    layer_2_genders = layers.Concatenate()([input_gender_1, input_gender_2])
    layer_2_dobs = layers.Concatenate()(
        [layer_1_input_dob_1, layer_1_input_dob_2])

    layer_3_names = layers.Dense(name_len, activation='relu')(layer_2_names)
    layer_3_genders = layers.Dense(
        2*gender_len, activation='relu')(layer_2_genders)
    layer_3_dobs = layers.Dense(2*dob_len, activation='relu')(layer_2_dobs)

    layer_4_combined = layers.Concatenate()(
        [layer_3_names, layer_3_genders, layer_3_dobs])
    layer_5_brain = layers.Dense(10, activation='relu')(layer_4_combined)
    layer_6_decider = layers.Dense(1, activation='sigmoid')(layer_5_brain)

    model = keras.Model(inputs=[
        input_name_1,
        input_gender_1,
        input_dob_1,
        input_name_2,
        input_gender_2,
        input_dob_2
    ], outputs=[
        layer_6_decider
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # print(model.summary())

    return model


def fuzzyChecker(seq1, seq2):
    chars = "".join(list(map(lambda x: chr(x), seq1)))
    chars2 = "".join(list(map(lambda x: chr(x), seq2)))
    return fuzz.ratio(chars, chars2)


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
    i_fuzz_n = keras.Input(name="fuzz_n", shape=(1, ))
    i_fuzz_g = keras.Input(name="fuzz_g", shape=(1, ))

    i_name_1_rs = layers.Rescaling(scale=1./61, offset=-1)(i_name_1)
    i_name_2_rs = layers.Rescaling(scale=1./61, offset=-1)(i_name_2)
    i_dob_1_rs = layers.Rescaling(scale=1./29, offset=-1)(i_dob_1)
    i_dob_2_rs = layers.Rescaling(scale=1./29, offset=-1)(i_dob_2)
    i_fuzz_n_rs = layers.Rescaling(scale=1./50, offset=-1)(i_fuzz_n)
    i_fuzz_g_rs = layers.Rescaling(scale=1./50, offset=-1)(i_fuzz_g)

    l1_name1 = layers.LocallyConnected1D(10, 3, activation='relu')(i_name_1_rs)
    l1_name2 = layers.LocallyConnected1D(10, 3, activation='relu')(i_name_2_rs)
    l1_dob1 = layers.LocallyConnected1D(5, 3, activation='relu')(i_dob_1_rs)
    l1_dob2 = layers.LocallyConnected1D(5, 3, activation='relu')(i_dob_2_rs)

    l2_names = layers.Concatenate()([l1_name1, l1_name2])
    l2_names = layers.Flatten()(l2_names)
    l2_genders = layers.Concatenate()([i_gender_1, i_gender_2])
    l2_dobs = layers.Concatenate()([l1_dob1, l1_dob2])
    l2_dobs = layers.Flatten()(l2_dobs)

    l3_names = layers.Dense(10, activation='relu')(l2_names)
    l3_genders = layers.Dense(10, activation='relu')(l2_genders)
    l3_dobs = layers.Dense(10, activation='relu')(l2_dobs)
    l3_fuzz_g = layers.Dense(10, activation='relu')(i_fuzz_n_rs)
    l3_fuzz_n = layers.Dense(10, activation='relu')(i_fuzz_g_rs)

    l4_combined = layers.Concatenate()(
        [l3_names, l3_genders, l3_dobs, l3_fuzz_g, l3_fuzz_n])
    l5_brain = layers.Dense(20, activation='relu')(l4_combined)
    l6_decider = layers.Dense(1, activation='sigmoid')(l5_brain)

    model = keras.Model(inputs=[
        i_name_1,
        i_gender_1,
        i_dob_1,
        i_name_2,
        i_gender_2,
        i_dob_2,
        i_fuzz_n,
        i_fuzz_g
    ], outputs=[
        l6_decider
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # print(model.summary())

    return model


def to_dict(seq):
    return {
        "name_1": np.asarray(list(map(lambda x: np.asarray(x[0]), seq))),
        "gender_1":  np.asarray(list(map(lambda x: np.asarray(x[1]), seq))),
        "dob_1":  np.asarray(list(map(lambda x: np.asarray(x[2]), seq))),
        "name_2":  np.asarray(list(map(lambda x: np.asarray(x[3]), seq))),
        "gender_2":  np.asarray(list(map(lambda x: np.asarray(x[4]), seq))),
        "dob_2":  np.asarray(list(map(lambda x: np.asarray(x[5]), seq))),
        "fuzz_n": np.asarray(list(map(lambda x: fuzzyChecker(x[0], x[3]), seq))),
        "fuzz_g": np.asarray(list(map(lambda x: fuzzyChecker(x[2], x[5]), seq)))
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
