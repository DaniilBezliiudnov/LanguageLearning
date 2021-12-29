import names.data_reader_csv as data_reader
import names.data_preparer as data
import names.model as model

(names, genders, dates) = data_reader.get_data('\\data\\names.csv')
staticData = data.prepare_data(names, len(names[0]), genders, dates)
(my_model, history) = model.train_model(staticData)
model.print_history(history)
