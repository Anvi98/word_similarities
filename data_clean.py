from all_imports import nltk, nlp, re, locate, spacy, np, stopwords

with open("text_acad.txt", 'r') as f:
    raw_f = f.readlines()

## Clean data
data_clean = []
unique_tokens = []

print("Cleaning the data...")
for sentence in raw_f:
    tmp_raw = []
    clean_pos = []
    # Remove Identifier in raw file which looks like => @@4000241
    tmp_raw = ' '.join(sentence.split()[1:])
    # POS tagging the current text
    tmp_raw = nltk.pos_tag(nltk.word_tokenize(tmp_raw))
    # Take only text not tags in tuple after POS
    tmp_raw = [ clean_pos.append(tup[0]) for tup in tmp_raw]
    # We convert into a string to be able to lemmatize
    clean_pos = ' '.join(clean_pos)
    # Use spacy to lemmatize
    tmp_lemma = nlp(clean_pos)
    tmp_lemma = [word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in tmp_lemma]
    # Remove stopwords from lemmatized version
    stops = set(stopwords.words('english'))
    tmp_stop = [ w.lower() for w in tmp_lemma if w.lower() not in stops]
    tmp_stop = ' '.join(tmp_stop)
    #Remove special characters
    tmp_r = re.sub('[^a-zA-Z]+', ' ', tmp_stop)
    #Remove words which length is less than 3
    tmp_clean = [ w for w in tmp_r.split() if len(w) > 3]
    # Create List of unique tokens => vocab
    for token in tmp_clean:
        if token not in unique_tokens:
            unique_tokens.append(token)
    # Append clean text to data_clean
    data_clean.append(tmp_clean)

# np.save("numpy_data/data_clean.npy", data_clean)
# np.save("numpy_data/unique_tokens.npy", unique_tokens)

## collect Context for each word in vocabulary
context_dict = {}
window= 8
print("Collecting contexts for each word...")
for word in unique_tokens:
    tmp_c = []
    for text in data_clean:
        # Find index of word in text
        w_indices = []
        if word in text:
            # If in current text the word appear many times, save the positions at which it appears and then generates the context
            if text.count(word) > 1:
                pred = lambda x:x == word
                w_indices = list(locate(text,pred= pred))
                # print(w_indices)
                for ind in w_indices:
                    # get number of words before and after word in text
                    win = int(window / 2)
                    # Get the context of word in current text
                    before = text[:ind][-win:]
                    after = text[(ind+1):][:win]
                    tmp_context = before + after
                    tmp_c.append(tmp_context)
            else:
                tmp_i = text.index(word)
                # get number of words before and after word in text
                win = int(window / 2)
                # Get the context of word in current text
                before = text[:tmp_i][-win:]
                after = text[(tmp_i+1):][:win]
                tmp_context = before + after
                tmp_c.append(tmp_context)
    
    context_dict[word] = tmp_c

context_list = list(context_dict.values())

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

np.save('numpy_data/unique_tokens.npy', unique_tokens)
context_list = list(context_dict.values())