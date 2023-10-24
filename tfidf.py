from all_imports import np, idf_list, tf

tfidf_list = np.multiply(tf, idf_list)

np.save('numpy_data/tfidf_list.npy', tfidf_list)