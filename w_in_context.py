from all_imports import raw_freq, np

print("Calculating the number of Contexts where each word appears...")
matrix_freq = np.matrix(raw_freq)
# If a frequency off a word is greater than 1 make it equal to 1.
matrix_freq[matrix_freq >= 1.0] = 1.0

# We need this information when calculating TFIDF
w_in_context = [] # Length is the size of vocab with only one dimension
for i in range(matrix_freq.shape[1]):
    # Sum values by column in matrix since each column represents the frequencies of a specific word
    tmp_num_context = np.sum(matrix_freq[:, i])
    w_in_context.append(tmp_num_context)

np.save('numpy_data/w_in_context.npy', w_in_context)