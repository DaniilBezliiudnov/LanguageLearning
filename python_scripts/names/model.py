from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from matplotlib import pyplot as plt
import numpy as np


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

def fuzzyChecker(x):
    print(x)
    return x


def create_model_v2(data):
    name_len = len(data['training_data'][0][0])
    gender_len = len(data['training_data'][0][1])
    dob_len = len(data['training_data'][0][2])

    input_name_1 = keras.Input(name="name_1", shape=(name_len, 1))
    input_name_2 = keras.Input(name="name_2", shape=(name_len, 1))
    input_gender_1 = keras.Input(name="gender_1", shape=(gender_len, ))
    input_gender_2 = keras.Input(name="gender_2", shape=(gender_len, ))
    input_dob_1 = keras.Input(name="dob_1", shape=(dob_len, 1))
    input_dob_2 = keras.Input(name="dob_2", shape=(dob_len, 1))

    input_name_1_rs = layers.Rescaling(scale=1./61, offset =-1)(input_name_1)
    input_name_2_rs = layers.Rescaling(scale=1./61, offset =-1)(input_name_2)
    
    input_dob_1_rs = layers.Rescaling(scale=1./29, offset =-1)(input_dob_1)
    input_dob_2_rs = layers.Rescaling(scale=1./29, offset =-1)(input_dob_2)
    
    layer_x_names = layers.Concatenate(axis = 0)([input_name_1, input_name_2])
    layer_x_names = layers.Flatten()(layer_x_names)
    layer_x2_names = layers.Lambda(fuzzyChecker)(layer_x_names)
    layer_x3_names = layers.Dense(2, activation='relu')(layer_x2_names)
    
    layer_1_input_name1 = layers.LocallyConnected1D(
        10, 3, activation='relu')(input_name_1_rs)
    layer_1_input_name2 = layers.LocallyConnected1D(
        10, 3, activation='relu')(input_name_2_rs)
    # layer_1_input_name_1 = layers.Dense(name_len, activation='relu')(input_name_1)
    # layer_1_input_name_2 = layers.Dense(name_len, activation='relu')(input_name_2)
    # layer_1_input_dob_1 = layers.Dense(dob_len, activation='relu')(input_dob_1)
    # layer_1_input_dob_2 = layers.Dense(dob_len, activation='relu')(input_dob_2)
    layer_1_input_dob1 = layers.LocallyConnected1D(
        5, 3, activation='relu')(input_dob_1_rs)
    layer_1_input_dob2 = layers.LocallyConnected1D(
        5, 3, activation='relu')(input_dob_2_rs)

    layer_2_names = layers.Concatenate()(
        [layer_1_input_name1, layer_1_input_name2])
    layer_2_names = layers.Flatten()(layer_2_names)
    layer_2_genders = layers.Concatenate()([input_gender_1, input_gender_2])
    layer_2_dobs = layers.Concatenate()(
        [layer_1_input_dob1, layer_1_input_dob2])
    layer_2_dobs = layers.Flatten()(layer_2_dobs)

    # layer_3_names = layers.Dense(60, activation='relu')(layer_2_names)
    # layer_3_genders = layers.Dense(20, activation='relu')(layer_2_genders)
    # layer_3_dobs = layers.Dense(80, activation='relu')(layer_2_dobs)

    layer_4_combined = layers.Concatenate()(
        [layer_2_names, layer_2_genders, layer_2_dobs, layer_x3_names])
    layer_5_brain = layers.Dense(100, activation='relu')(layer_4_combined)
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


def to_dict(seq):
    return {
        "name_1": np.asarray(list(map(lambda x: np.asarray(x[0]), seq))),
        "gender_1":  np.asarray(list(map(lambda x: np.asarray(x[1]), seq))),
        "dob_1":  np.asarray(list(map(lambda x: np.asarray(x[2]), seq))),
        "name_2":  np.asarray(list(map(lambda x: np.asarray(x[3]), seq))),
        "gender_2":  np.asarray(list(map(lambda x: np.asarray(x[4]), seq))),
        "dob_2":  np.asarray(list(map(lambda x: np.asarray(x[5]), seq)))
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
