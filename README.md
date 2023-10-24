
# Compute Word Similarities - From Scratch using Python3.

## Learning objectives

- Learn how to compute basic fixed contexts of words in a corpus.
- Learn how to generate Raw frequencies, Relative Frequencies, Term-Frequencies (TF), Inverse-Document-Frequency (IDF), TF-IDF and Finally BM25 Okapi of Terms from scratch without using fancy libraries (Only Numpy is necessary for that)
- Lear Dynamic programming
- Get insights on Their related performances.
- A pre-work before understanding how LLMs perform tokenizations and extract contexts.


## Description

In this repository, I implement an algorithm that:
- Clean a raw Corpus 
- Compute several type of frequencies as features
- Compute Similarities via Dot Product and Scaled Dot Product
- Generate Files containing similarities of chosen words. (Which are in the generated vocabulary)

### Dataset
The Corpus I used in this task, is the [COCA]('https://www.english-corpora.org/coca/') academic dataset. It contains 265 texts related to academic. After preprocessing
the corpus I was left with roughly 36K tokens. You can get the dataset also from this repository. It is referred as 'text_acad.txt'.

### Workflow
The execution of the code can be quite resource-intensive, depending on the machine and how it's configured. In my particular environment, it took an entire hour to generate all the required frequencies. To make this possible, I had to divide certain calculations into separate scripts since my system lacked the necessary memory to run everything in a single file. These scripts included:

    -data_clean.py
    -raw_freq.py
    -w_in_context.py
    -relative_freq.py
    -tf.py
    -idf.py
    -tfidf.py
One critical aspect was managing the all_imports.py file, as it is shared among all the Python scripts mentioned above. At each stage, I had to carefully comment and uncomment the imports related to Numpy data in the .npy format. This might seem like an unusual approach, but it was the workaround I found to successfully run the code on my laptop. Unfortunately, running all the computations in a single Python file was not feasible due to the limitations of my system.

### Task requirements

- This task does require very few libraries. There are: Numpy, NLTK and spacy. But the code also needs to be runned in a virtual environment. Just create one and install the required packages via:
```
pip3 install -r requirements.txt

```

### Author

👤 **Alex Eponon**
​
- GitHub: [@Anvi98](https://github.com/Anvi98)

- Twitter: [@anvi_al](https://twitter.com/anvi_al)

- LinkedIn: [Alex Eponon](https://www.linkedin.com/in/anvi-alex-eponon/)

