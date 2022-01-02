from pipe import where
import names.model as model
import names.data_preparer as data

res = model.model.predict(data.test_data)
res_bool = [1 if x > 0.5 else 0 for x in res]
l_res_bol = len(res_bool)
correct_answers = len(list(zip(res_bool, data.test_labels) | where(lambda x: x[0] == x[1])))
percent = round(correct_answers*1.0/l_res_bol, 3)

# VISUALIZATION

print(f'I have {correct_answers} correct answers out of {l_res_bol} - {percent}%')
print(res)
