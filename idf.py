from all_imports import np, w_in_context, raw_freq

print("Computing IDF of terms...")
idf_list = np.log(raw_freq / w_in_context + 1)

np.save("numpy_data/idf_list.npy", idf_list)