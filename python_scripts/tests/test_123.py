from pipe import select
from fuzzywuzzy import fuzz

seq = [ 97, 105, 100, 101, 110]

chars = list(seq | select(lambda x: chr(x)))
print (chars)
# content of test_sample.py
def inc(x_val):
    return x_val + 1


def test_answer():
    assert inc(4) == 5
    