from all_imports import raw_freq, np

relative_freq = []
print("Compute relative frequences out of raw frequencies...")
for i in range(raw_freq.shape[1]):
    scalar_tmp_matrix_raw_freq = np.sum(np.matrix(raw_freq[:, i]))
    tmp_column_rfreq = np.matrix(raw_freq[:, i]) / scalar_tmp_matrix_raw_freq
    relative_freq.append(tmp_column_rfreq)
relative_freq = np.array(relative_freq)

np.save("numpy_data/rel_freq.npy", relative_freq)