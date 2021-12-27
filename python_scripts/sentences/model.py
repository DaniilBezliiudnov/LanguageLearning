import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import data_preparer as data


def print_history(history):
    plt.plot(history.epoch, history.history['accuracy'])
    plt.plot(history.epoch, history.history['loss'])
    plt.legend(["accuracy", "loss"])
    plt.title("accuracy/loss function")
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("accuracy/loss")
    plt.show()


# MODEL CREATION
model = keras.Sequential([
    layers.Embedding(data.dic_size, 10),
    layers.GlobalAveragePooling1D(),
    layers.Dense(24, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

#adam = keras.optimizers.Adam(learning_rate=0.0001,
#                             beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# MODEL TRAINING
trainingHistory = model.fit(data.training_seq,
                            data.training_labels,
                            epochs=200,
                            verbose=0)

res = model.predict(data.test_seq)
print(res)
# VISUALIZATION
# print(model.summary())
print_history(trainingHistory)
