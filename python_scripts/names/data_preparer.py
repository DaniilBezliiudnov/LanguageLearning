import numpy as np
from pipe import select
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences


def to_normalized_int_sequence(seq):
    padded_int_seq = pad_sequences(
        list(seq | select(lambda x:
                          list(x | select(lambda y: ord(y))))))
    normalized_padded_int_seq = [x/122.0 for x in padded_int_seq]
    return normalized_padded_int_seq

def date_to_normalized_int_sequence(seq):
    return list(seq | select(lambda x: [x.year/2022., x.month/12., x.day/31.]))

def prepare_data(names, genders, dates):
    
    name_seq = to_normalized_int_sequence(names)
    gender_seq = to_normalized_int_sequence(genders)
    date_seq = date_to_normalized_int_sequence(dates)
    data_seq = list(
        map(lambda x, y, z: [*x, *y, *z], name_seq, gender_seq, date_seq))

    rev_data_seq = data_seq.copy()
    rev_data_seq.reverse()

    true_seq = list(map(lambda x, y: (1, [*x, *y]), data_seq, data_seq))
    false_seq = list(map(lambda x, y: (0, [*x, *y]), data_seq, rev_data_seq))
    rnd_seq = list(true_seq.copy()
                | select(lambda x:
                            (
                                x[0],
                                [random.randint(0, 122)/122.0 if i == random.randint(
                                    1, len(x[1])) else el for i, el in enumerate(x[1])]
                            )))


    total_seq = true_seq + false_seq + rnd_seq
    random.shuffle(total_seq)

    training_data_size = int(len(total_seq) * 50 / 100)
    total_data_size = int(len(total_seq) * 100 / 100)

    training_data = np.asarray(
        list(total_seq[:training_data_size] | select(lambda x: x[1])))
    training_labels = np.asarray(
        list(total_seq[:training_data_size] | select(lambda x: x[0])))
    test_data = np.asarray(
        list(total_seq[training_data_size:total_data_size]
            | select(lambda x: x[1])))
    test_labels = np.array(
        list(total_seq[training_data_size:total_data_size] | select(lambda x: x[0])))
    # VISUALIZATION
    print(f'training seq {len(training_data)}:')
    print(training_data[0])
    print(f'test seq {len(test_data)}:')
    print(test_data[0])
    return {
        'training_data' : training_data,
        'training_labels' : training_labels,
        'test_data' : test_data,
        'test_labels' : test_labels
    }