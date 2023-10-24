from all_imports import np, raw_freq

tf = np.array(raw_freq) / np.array(raw_freq).shape[1]

np.save('numpy_data/tf.npy', tf)