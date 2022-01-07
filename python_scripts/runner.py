from names import config
import names.data_reader_csv as data_reader
from names import data_preparer
from names import model
from predicter import predict


def run_predict(my_model, names, genders, dates):
    normalized_data = data_preparer.normalize_merge_data(
        names, len(names[0]), genders, dates)

    for i, x in enumerate(normalized_data):

        prediction_data = list(map(
            lambda s, i=i: [normalized_data[i][0],
                            normalized_data[i][1],
                            normalized_data[i][2],
                            s[0], s[1], s[2]],
            normalized_data[i+1:]))

        indicies = predict(my_model, prediction_data)
        dups = list(map(lambda j, i=i:
                        f'{j[1]:.2} : {names[i]}-{genders[i]}-{dates[i]}' +
                        f' | {names[j[0]]}-{genders[j[0]]}-{dates[j[0]]}',
                        indicies))
        print('\n'.join(dups))


def run(names, genders, dates, validNames=[], validGender=[], validDates=[]):
    if len(validNames) == 0:
        validNames = names
    if len(validGender) == 0:
        validGender = genders
    if len(validDates) == 0:
        validDates = dates

    train_data = data_preparer.prepare_data(
        names, len(names[0]), genders, dates, 100)
    my_model = model.create_model_v2(train_data)
    history = model.train_model(my_model, train_data, 80)

    normalized_data = data_preparer.normalize_merge_data(
        names, len(names[0]), genders, dates)
    return my_model


(names, genders, dates) = data_reader.get_data(
    config.root_dir + '\\data\\names.csv')
m = run(names, genders, dates)
run_predict(m, names, genders, dates)
