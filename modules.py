from numba import jit
import spacy
import numpy as np

@jit(cache=True,forceobj=True)
def getTf(raw_freq):
    tf = np.array(raw_freq) / np.array(raw_freq).shape[1]
    return tf

@jit(cache=True, forceobj=True)
def getIdf(raw_freq, w_in_context):
    idf_list = np.log(np.array(raw_freq) / np.array(w_in_context) + 1)
    return idf_list

def dot_product(input, unique_tokens, features):
    i = 0
    results = []
    for k, feature in features.items():
        print(f"Calculating Dot product of input word with other words - feature : {k}...")
        if input in unique_tokens:
            i = unique_tokens.index(input)
        W_input = feature[i]
        sim_input = []
        W_input = W_input.reshape(-1,1)
        for j in range(feature.shape[0]):
            word = unique_tokens[j]
            if k == 'relative frequence':
                tmp = np.dot(W_input.T, np.array(feature[j]).T)
            else:
                tmp = np.dot(W_input.T, np.array(feature[j]))
            sim_input.append([tmp[0], word])
        results.append(sim_input)
    return results

@jit(forceobj=True)
def scaled_dot_product(input, unique_tokens, features):
    i=0
    results = []
    for k, feature in features.items():
        
        print(f"Calculating Scaled Dot product of input word with other words - feature : {k}...")
        if input in list(unique_tokens):
            i = list(unique_tokens).index(input)
        W_input = feature[i]
        sim_input = []
        W_input = W_input.reshape(-1,1)
        for j in range(feature.shape[0]):
            word = unique_tokens[j]
            if k == 'relative frequence':
                tmp = np.dot(W_input.T, np.array(feature[j]).T) / (len(feature[i]) * len(feature[j]))
            else:
                tmp = np.dot(W_input.T, np.array(feature[j])) / (len(feature[i]) * len(feature[j]))
            sim_input.append([tmp[0], word])
        results.append(sim_input)
    print("End of Calculation...")
    return results

# def similarityResults(sim_tf, sim_idf, sim_tfidf):
#     print("Generate sorted similarities... ")
#     sim_tf = list(sorted(sim_tf, reverse=True))
#     sim_idf = list(sorted(sim_idf, reverse=True))
#     sim_tfidf = list(sorted(sim_tfidf, reverse=True))

#     return sim_tf, sim_idf, sim_tfidf

def similarityResults(sim_relative):
    print("Generate sorted similarities... ")
    results = []
    for sim in sim_relative:
        tmp = list(sorted(sim, reverse=True))
        results.append(tmp)
    # sim_relative = list(sorted(sim_relative, reverse=True))

    return results

def saveFile(data, input):
    for k, name in data.items():
        with open(f"{input}_{k}.txt", 'w') as f:
            f.write(str(name[:1000]))