from names import config
import names.data_reader_csv as data_reader
from names import data_preparer
from names import model
from predicter import predict


def run_predict(my_model, n_len, names, genders, dates):
    normalized_data = data_preparer.normalize_merge_data(
        names, n_len, genders, dates)

    for i_x, x in enumerate(normalized_data):

        prediction_data = list(map(
            lambda d, i=i_x: [normalized_data[i][0],
                              normalized_data[i][1],
                              normalized_data[i][2],
                              d[0], d[1], d[2]],
            normalized_data[i_x+1:]))

        indicies = predict(my_model, prediction_data)
        dups = list(map(lambda i, j=i_x:
                        f'{i[1]:.2} : {names[j]}-{genders[j]}-{dates[j]}' +
                        f' | {names[i[0]]}-{genders[i[0]]}-{dates[i[0]]}',
                        indicies))
        print('\n'.join(dups))


def run(names, n_len, genders, dates):
    train_data = data_preparer.prepare_data(
        names, n_len, genders, dates, 100)
    my_model = model.create_model_v2(train_data)
    history = model.train_model(my_model, train_data, 80)
    return my_model


(names, genders, dates) = data_reader.get_data(
    config.root_dir + '\\data\\names.csv')
m = run(names, 15, genders, dates)
run_predict(m, 15, names, genders, dates)
