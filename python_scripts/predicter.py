from pipe import where, select
from tensorflow.keras import Model
from names.model import to_dict


def predict(model: Model, prediction_data):
    res = model.predict(to_dict(prediction_data))

    result = [(i, x[0]) if x > 0.5 else (-1, -1)
              for i, x in enumerate(res)]
    result = list(result | where(lambda x: x[0] >= 0))

    l_res = len(result)

    # VISUALIZATION
    print(f'I have found the following amount of dups {l_res}')
    return result
