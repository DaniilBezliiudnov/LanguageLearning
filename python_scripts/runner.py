from names import config
import names.data_reader_csv as data_reader
import names.data_preparer as data_preparer
from names import model
from predicter import predict

(names, genders, dates) = data_reader.get_data(
    config.root_dir + '\\data\\names.csv')
train_data = data_preparer.prepare_data(
    names, len(names[0]), genders, dates, 100)
my_model = model.create_model_v2(train_data)
history = model.train_model(my_model, train_data, 80)

normalized_data = data_preparer.normalize_merge_data(
    names, len(names[0]), genders, dates)

for i, x in enumerate(normalized_data):
    prediction_data = list(map(
        lambda x: [normalized_data[i][0],
                normalized_data[i][1],
                normalized_data[i][2],
                x[0], x[1], x[2]],
        normalized_data))

    indicies = predict(my_model, prediction_data)
    dups = list(map(
        lambda j: f'{j[1]:.2} : {names[i]}-{genders[i]}-{dates[i]} | {names[j[0]]}-{genders[j[0]]}-{dates[j[0]]}',
        indicies))
    print('\n'.join(dups))
# model.print_history(history)
