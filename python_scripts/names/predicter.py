from pipe import where
import names.model as model
import names.data_preparer as data

res = model.model.predict(data.test_data)
res_bool = [1 if x > 0.5 else 0 for x in res]
correct_answers = len(list(zip(res_bool, data.test_labels) | where(lambda x: x[0] == x[1])))

# VISUALIZATION
print(f'I have {correct_answers} correct answers out of {len(res_bool)} - {round(correct_answers*1.0/len(res_bool), 3)}%')
print(res)