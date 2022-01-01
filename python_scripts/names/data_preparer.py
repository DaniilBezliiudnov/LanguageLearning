import numpy as np
from pipe import select
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences


def to_normalized_int_sequence(seq, padding_max_len):
    padded_int_seq = pad_sequences(
        list(seq | select(lambda x:
                          list(x | select(lambda y: ord(y))))), maxlen=padding_max_len)
    normalized_padded_int_seq = [x/122.0 for x in padded_int_seq]
    return normalized_padded_int_seq


def date_to_normalized_int_sequence(seq):
    return list(seq | select(lambda x:
                             [x.year/2022., x.month/12., x.day/31.]))


def normalize_merge_data(names, names_max_len, genders, dates):
    name_seq = to_normalized_int_sequence(names, names_max_len)
    gender_seq = to_normalized_int_sequence(genders, 4)
    date_seq = date_to_normalized_int_sequence(dates)
    return list(map(lambda x, y, z: (x, y, z), name_seq, gender_seq, date_seq))


def randomize_list(l: list):
    l_new = l.copy()
    li = random.randint(0, len(l_new)-1)
    l_new[li] = l_new[li] + (random.random() - 1) * 0.1
    return l_new


def randomize(x):
    return (randomize_list(x[0]), randomize_list(x[1]), randomize_list(x[2]))


def prepare_seq(seq_a, seq_b, label):
    return list(
        map(lambda x, y: (label, [x[0], x[1], x[2], y[0], y[1], y[2]]), seq_a, seq_b))


def prepare_data(names, names_max_len, genders, dates, total_data_percent):
    data_seq = normalize_merge_data(names, names_max_len, genders, dates)

    rev_data_seq = data_seq.copy()
    rev_data_seq.reverse()

    shifted_data_seq = data_seq.copy()
    shifted_data_seq.insert(0, shifted_data_seq.pop())

    true_seq = prepare_seq(data_seq, data_seq, 1)
    rnd_seq = prepare_seq(data_seq, (data_seq | select(randomize)), 1)
    rnd_seq2 = prepare_seq(data_seq, (data_seq | select(randomize)), 1)
    false_seq = prepare_seq(data_seq, rev_data_seq, 0)
    shifted_seq = prepare_seq(data_seq, shifted_data_seq, 1)

    total_seq = true_seq + false_seq + shifted_seq + rnd_seq + rnd_seq2
    random.shuffle(total_seq)

    total_data_size = int(len(total_seq) * total_data_percent / 100)
    training_data_size = int(total_data_size * 60 / 100)

    training_data = np.asarray(
        list(total_seq[:training_data_size] | select(lambda x: x[1])))
    training_labels = np.asarray(
        list(total_seq[:training_data_size] | select(lambda x: x[0])))
    test_data = np.asarray(
        list(total_seq[training_data_size:total_data_size]
             | select(lambda x: x[1])))
    test_labels = np.array(
        list(total_seq[training_data_size:total_data_size]
             | select(lambda x: x[0])))
    # VISUALIZATION
    print(f'training seq {len(training_data)}:')
    print(training_data[0])
    print(f'test seq {len(test_data)}:')
    print(test_data[0])
    return {
        'training_data': training_data,
        'training_labels': training_labels,
        'test_data': test_data,
        'test_labels': test_labels
    }
