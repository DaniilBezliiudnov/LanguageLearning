import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt


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

    adam = keras.optimizers.Adam(learning_rate=0.0001,
                                beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='binary_crossentropy',
                optimizer=adam, metrics=['accuracy'])
    
    print(model.summary())
    
    return model

def train_model(model : keras.Sequential, data):
    trainingHistory = model.fit(
                                data['training_data'],
                                data['training_labels'],
                                batch_size=10,
                                epochs=50,
                                validation_data=(data['test_data'], data['test_labels'])
                                )

    return trainingHistory
