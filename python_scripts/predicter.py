from pipe import where
from tensorflow.keras import Model
from names.model import to_dict


def predict(model: Model, prediction_data):
    res = model.predict(to_dict(prediction_data))

    result = [(i, x[0]) if x > 0.65 else (-1, -1)
              for i, x in enumerate(res)]
    result = list(result | where(lambda x: x[0] >= 0))
    result.sort(key=lambda x: x[1], reverse=True)

    # VISUALIZATION
    # l_res = len(result)

    # res = list(map(
    #     lambda x: int(x*100),
    #     res
    # ))
    # res = [[y for y in res if y <= x + 5 and y > x - 5]
    #        for x in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
    # res = list(map(
    #     lambda x: len(x),
    #     res
    # ))

    # print(res)
    # print(f'I have found the following amount of dups {l_res}')
    return result
