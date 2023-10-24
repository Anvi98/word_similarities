from all_imports import *

if __name__ == '__main__':

        inputs = ['student', 'review', 'director']

        for input in inputs:

                features = {'term-frequencies':tf, 'IDF':idf_list, 'tfIdf':tfidf_list}

                ## Scaled Dot Product of TF, IDF, TF-IDF features
                results = scaled_dot_product(input, unique_tokens, features)

                # sc_tf, sc_idf, sc_tfidf = results
                sc_tf, sc_idf, sc_tfidf = similarityResults(results)
                sc_dot = {
                        'term-frequencies_with_Scaled_dot':sc_tf, 
                        'IDF_with_Scaled_dot':sc_idf, 
                        'tfIdf_with_Scaled_dot': sc_tfidf
                        }
                # Save to file
                saveFile(sc_dot, input)

                # Normal Dot Product TF, IDF, TF-IDF features
                results = dot_product(input, list(unique_tokens), features)
                sim_tf, sim_idf, sim_tfidf = similarityResults(results)

                sims_dot = {
                        'term-frequencies_with_dot':sim_tf, 
                        'IDF_with_dot':sim_idf, 
                        'tfIdf_with_dot': sim_tfidf
                        }
                saveFile(sims_dot, input)

                ##---------Relative frequencies features-------------##

                features = { 'relative frequence': rel_freq}

                ## Scaled Dot prdouct of Relative Frequencies
                results = scaled_dot_product(input, unique_tokens, features)

                sc_rel = similarityResults(results)
                sc_dot = {
                        'relative_frequencies_with_Scaled_dot':sc_rel, 
                        }

                # Save to file
                saveFile(sc_dot, input)

                # Normal Dot Product Relative Frequencies Features
                results = dot_product(input, list(unique_tokens), features)
                sim_rel= similarityResults(results)

                sims_dot = {
                        'relative_frequencies_with_dot':sim_rel, 
                        }

                saveFile(sims_dot, input)