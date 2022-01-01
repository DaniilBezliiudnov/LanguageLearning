import names.config as config
import names.data_reader_csv as data_reader
import names.data_preparer as data
import names.model as model

(names, genders, dates) = data_reader.get_data(config.root_dir + '\\data\\names.csv')
staticData = data.prepare_data(names, len(names[0]), genders, dates, 50)
my_model = model.create_model_v2(staticData)
history = model.train_model(my_model, staticData, 20)
# model.print_history(history)
