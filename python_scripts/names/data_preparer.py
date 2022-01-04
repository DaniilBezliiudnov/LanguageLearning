import random
import numpy as np
from pipe import select
from tensorflow.keras.preprocessing.sequence import pad_sequences


def to_normalized_int_sequence(seq, padding_max_len, norm_coeff):
    padded_int_seq = pad_sequences(
        list(seq | select(lambda x:
                          list(x | select(lambda y: ord(y))))), maxlen=padding_max_len)
    normalized_padded_int_seq = [x/norm_coeff for x in padded_int_seq]
    return normalized_padded_int_seq


def date_to_str_sequence(seq):
    return list(seq | select(lambda x: x.strftime("%Y%m%d")))


def normalize_merge_data(names, names_max_len, genders, dates):
    name_seq = to_normalized_int_sequence(names, names_max_len, 122.0)
    gender_seq = to_normalized_int_sequence(genders, 1, 122.0)
    date_seq = date_to_str_sequence(dates)
    date_seq = to_normalized_int_sequence(date_seq, 8, 58.0)
    return list(map(lambda x, y, z: (x, y, z), name_seq, gender_seq, date_seq))


def randomize_list(list_to_sort: list):
    l_new = list_to_sort.copy()
    l_index = random.randint(0, len(l_new)-1)
    l_new[l_index] = l_new[l_index] + (random.random() - 1) * 0.001
    return l_new


def randomize(list_of_lists):
    return (
        randomize_list(list_of_lists[0]),
        list_of_lists[1],
        randomize_list(list_of_lists[2])
    )


def prepare_seq(seq_a, seq_b, label):
    return list(
        map(lambda x, y: (label, [x[0], x[1], x[2], y[0], y[1], y[2]]), seq_a, seq_b))


def finalize_data_preparation(total_seq, total_data_percent):
    total_data_size = int(len(total_seq) * total_data_percent / 100)
    training_data_size = int(total_data_size * 60 / 100)

    random.shuffle(total_seq)

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


def prepare_data(names, names_max_len, genders, dates, total_data_percent):
    data_seq = normalize_merge_data(names, names_max_len, genders, dates)

    rev_data_seq = data_seq.copy()
    rev_data_seq.reverse()

    shifted_data_seq = data_seq.copy()
    shifted_data_seq.insert(0, shifted_data_seq.pop())

    true_seq = prepare_seq(data_seq, data_seq, 1)
    rnd_seq = prepare_seq(data_seq, (data_seq | select(randomize)), 1)
    rnd_seq2 = prepare_seq(data_seq, (data_seq | select(randomize)), 1)
    rnd_seq3 = prepare_seq(data_seq, (data_seq | select(randomize)), 1)
    rnd_seq4 = prepare_seq(data_seq, (data_seq | select(randomize)), 1)
    false_seq = prepare_seq(data_seq, rev_data_seq, 0)
    shifted_seq = prepare_seq(data_seq, shifted_data_seq, 0)

    total_seq = true_seq + false_seq + rnd_seq + shifted_seq + \
        rnd_seq + rnd_seq2 + rnd_seq3 + rnd_seq4

    return finalize_data_preparation(total_seq, total_data_percent)
