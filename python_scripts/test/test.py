import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))

print(list(filter(lambda x: not x, [True, False, True])))
a = [x for x in range(1, 10, 1)]
print(a)
