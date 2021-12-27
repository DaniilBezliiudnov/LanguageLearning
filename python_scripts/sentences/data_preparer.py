import numpy as np
import data_reader_csv as data_reader
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

half_data_size = int(len(data_reader.data)/2)
training_data = data_reader.data[:half_data_size]
training_labels = np.asarray(data_reader.labels[half_data_size:]+data_reader.labels[half_data_size:])
test_data = data_reader.data[half_data_size:]
test_labels = data_reader.labels[half_data_size:]
# DATA PREPARATION

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(training_data)
dic_size = len(tokenizer.index_word)+1

training_seq = tokenizer.texts_to_sequences(training_data)
test_seq = tokenizer.texts_to_sequences(test_data)
max_len = max(
    max(len(x) for x in training_seq),
    max(len(x) for x in test_seq))

training_seq = np.concatenate((
    pad_sequences(training_seq, max_len),
    pad_sequences(training_seq, max_len, padding='post')))
test_seq = pad_sequences(test_seq, max_len)


# VISUALIZATION
print('word index: ', end='')
print(tokenizer.word_index)
print('training seq:')
print(training_seq)
print('test seq:')
print(test_seq)
