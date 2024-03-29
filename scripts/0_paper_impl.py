import time

import pandas as pd
import pickle as pkl
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# POT needs to be installed for the following code to work
# !pip install POT

df = pd.read_csv('../data/user-ct-test-collection-02-with-query-frequencies-and-tokens.txt', sep='\t')

frequencies = pd.read_csv('../data/query-frequencies-precomputed.txt', sep='\t', index_col=0)
tokens = pd.read_csv('../data/query-tokens-precomputed.txt', sep='\t', index_col=0)

# IDF scores of words
idf_scores = pd.read_csv('../data/idf-scores-precomputed.txt', sep='\t', index_col=0)
idf_scores = idf_scores['IDF'].to_dict()

# --------------------------------------------------------------------
# Tokenize queries (with tqdm progress bar)
queries = df['Query']
query_tokens = df['QueryTokens']

# Initialize word2vec model
AOL_model = Word2Vec(vector_size=200,  # Dimension of the word vectors
                     window=5,  # Context window size
                     min_count=1,  # Minimum word frequency
                     workers=4,  # Number of parallel workers
                     sg=1)  # Skip-gram model

# Train word2vec model
AOL_model.build_vocab(tqdm(query_tokens, desc='Building word2vec vocabulary'))

AOL_model.train(tqdm(query_tokens, desc='Training word2vec model'),
            total_examples=AOL_model.corpus_count,
            epochs=30)

# Save word2vec model
AOL_model.save('word2vec_AOL.model')
# --------------------------------------------------------------------


# Load Google's pre-trained Word2Vec model
# google_model = KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin', binary=True)  # if you have it locally
google_model = api.load('word2vec-google-news-300')

# Load the trained AOL word2vec model
# AOL_model = Word2Vec.load('../models/word2vec_AOL.model')  # Only if you don't want to retrain it (but it's quite fast to train, ~15 min)


# Query auto-completion function
def query_completion(query, completion_list, session_queries, alpha=0.6):
    N = len(session_queries)
    omega = 0.5

    # Combined score (alpha * similarity_score + (1-alpha) * frequency_score)
    combined_score = [frequencies.loc[a, 'Frequency'] if a in frequencies.index else 0 for a in completion_list] # Frequency score for now

    for i, candidate_query in enumerate(completion_list): # Candidate queries
        candidate_score = 0

        for session_query in session_queries:  # User session queries
            # Similarity scores:
            google_model_score = 0
            AOL_model_score = 0

            c_tokens = word_tokenize(candidate_query)
            x_tokens = word_tokenize(session_query)

            # Add similarities of all word pairs between the candidate query and the session query,
            # weighted by the idf of the words
            for c in c_tokens:
                for x in x_tokens:
                    try:
                        google_model_similarity = google_model.similarity(c, x)
                    except KeyError:
                        google_model_similarity = 0
                    try:
                        AOL_model_similarity = AOL_model.wv.similarity(c, x)
                    except KeyError:
                        AOL_model_similarity = 0

                    # Idf weighting average:
                    if c in idf_scores and x in idf_scores:
                        google_model_score += google_model_similarity * idf_scores[c] * idf_scores[x]
                        AOL_model_score += AOL_model_similarity * idf_scores[c] * idf_scores[x]
                    else:
                        google_model_score += google_model_similarity
                        AOL_model_score += AOL_model_similarity

            # Average, divide by all combinations of words:
            google_model_score /= (len(c_tokens) * len(x_tokens))
            AOL_model_score /= (len(c_tokens) * len(x_tokens))

            # Some query words might not appear in the google model's vocabulary:
            if google_model_score == 0:
                candidate_score += AOL_model_score
            else:
                candidate_score += omega * AOL_model_score + (1 - omega) * google_model_score

        candidate_score = alpha * candidate_score + (1 - alpha) * combined_score[i]
        combined_score[i] = candidate_score

    # Re-rank completion list
    re_ranked_list = [x for _, x in sorted(zip(combined_score, completion_list), reverse=True)]

    return re_ranked_list


# --------------------------------------------------------------------
# Train and test datasets (arrays of query sessions, sorted by time increasing)
with open('../data/train.pkl', 'rb') as f:
    train = pkl.load(f)

with open('../data/test.pkl', 'rb') as f:
    test = pkl.load(f)

# Flatten train dataset
train = [query for session in train for query in session]
train = pd.DataFrame(train, columns=['Query'])

# --------------------------------------------------------------------
# Parallelize the evaluation function

def evaluate_session(session_data):
    session, prefix_length, train_df, alpha = session_data

    # Last query in the session is the query we want to complete
    query_ground_truth = session[-1]

    # Skip if the query is shorter than the prefix length
    if len(query_ground_truth) < prefix_length:
        return 0

    # Queries preceding the last query in the session
    previous_queries = session[:-1] if len(session) > 1 else []

    # Get prefix (part of the query that needs to be completed)
    query_prefix = query_ground_truth[:prefix_length]

    # Get completions
    completions = train_df[train_df['Query'].str.startswith(query_prefix)]['Query'].unique()

    # Re-rank completions
    ranked_completions = query_completion(query_prefix, completions, previous_queries, alpha)

    try:
        RR_score = 1.0 / (ranked_completions.index(query_ground_truth) + 1)
    except ValueError:
        RR_score = 0.0

    return RR_score


def evaluate_test_set_parallel(sessions, prefix_length, train_df, alpha=0.6, num_processes=4):
    timeout_seconds = 8 * 3600  # timeout at 8 hours
    start_time = time.time()

    results = Parallel(n_jobs=num_processes)(delayed(evaluate_session)((session, prefix_length, train_df, alpha))
                                             for session in tqdm(sessions, desc='Evaluating completions prefix ' + str(prefix_length))
                                                if time.time() - start_time < timeout_seconds)

    print("Timeout after", (time.time() - start_time) / 3600, "hours")
    print("Sessions evaluated:", len(results), "/", len(sessions))
    return np.mean(results), len(results)


# Evaluate the test set
prefix_length = 5  # query will be completed based on the first 5 characters
mrr, num_sessions_evaluated = evaluate_test_set_parallel(test, prefix_length, train, alpha=0.6)
print("MRR for prefix length", prefix_length, ":", mrr)
with open('../results/word2vec_prefix5.txt', 'w') as f: f.write(str(mrr) + "\n" + "Sessions evaluated: " + str(num_sessions_evaluated))

prefix_length = 4  # query will be completed based on the first 4 characters
mrr, num_sessions_evaluated = evaluate_test_set_parallel(test, prefix_length, train, alpha=0.6)
print("MRR for prefix length", prefix_length, ":", mrr)
with open('../results/word2vec_prefix4.txt', 'w') as f: f.write(str(mrr) + "\n" + "Sessions evaluated: " + str(num_sessions_evaluated))

prefix_length = 3  # query will be completed based on the first 3 characters
mrr, num_sessions_evaluated = evaluate_test_set_parallel(test, prefix_length, train, alpha=0.6)
print("MRR for prefix length", prefix_length, ":", mrr)
with open('../results/word2vec_prefix3.txt', 'w') as f: f.write(str(mrr) + "\n" + "Sessions evaluated: " + str(num_sessions_evaluated))

prefix_length = 2  # query will be completed based on the first 2 characters
mrr, num_sessions_evaluated = evaluate_test_set_parallel(test, prefix_length, train, alpha=0.6)
print("MRR for prefix length", prefix_length, ":", mrr)
with open('../results/word2vec_prefix2.txt', 'w') as f: f.write(str(mrr) + "\n" + "Sessions evaluated: " + str(num_sessions_evaluated))

prefix_length = 1  # query will be completed based on the first character
mrr, num_sessions_evaluated = evaluate_test_set_parallel(test, prefix_length, train, alpha=0.6)
print("MRR for prefix length", prefix_length, ":", mrr)
with open('../results/word2vec_prefix1.txt', 'w') as f: f.write(str(mrr) + "\n" + "Sessions evaluated: " + str(num_sessions_evaluated))
