from all_imports import np
from data_clean import context_list, unique_tokens


raw_freq = []
print("Get raw frequencies of word in each context...")
for i in range(len(context_list)):
    tmp_freq_context = []
    for word in unique_tokens:
        tmp_count = 0
        for token_list in context_list[i]:
            if word in token_list:
                tmp_count += token_list.count(word)
        tmp_freq_context.append(tmp_count)
    raw_freq.append(tmp_freq_context)

print('save raw freq into numpy datafile...')
np.save("numpy_data/raw_freq.npy", raw_freq)
print("Raw_freq Done.")