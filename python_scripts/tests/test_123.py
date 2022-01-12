from pipe import where


def filter_positive_result(res):
    result = [(i, x[0]) if x[0] > 0.65 else (-1, -1)
              for i, x in enumerate(res)]
    result = list(result | where(lambda x: x[0] >= 0))
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def test_answer():
    actual = filter_positive_result(
        [[0], [1], [0.5], [0.6], [1], [0.12], [0.66], [0.42]])
    assert [(1, 1), (4, 1), (6, 0.66)] == actual
