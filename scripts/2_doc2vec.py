import time

import pandas as pd
import pickle as pkl
import gensim
from gensim.models import Doc2Vec
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
query_tokens = df['QueryTokens']  # list of tokenized queries


def tagged_queries(tokenized_queries):
    # Tag queries with unique IDs.
    for i, q_tokens in enumerate(tokenized_queries):
        yield gensim.models.doc2vec.TaggedDocument(q_tokens, [i])


# Train doc2vec model, on query sentences
model = Doc2Vec(vector_size=300, window=5, min_count=1, workers=4, epochs=30)

# Build vocabulary
model.build_vocab(tqdm(tagged_queries(query_tokens), desc='Building doc2vec vocabulary'))

# Train the model
model.train(tqdm(tagged_queries(query_tokens), desc='Training doc2vec model'),
            total_examples=model.corpus_count,
            epochs=model.epochs)

# Save the model
model.save('models/Doc2Vec_AOL.model')

# model = Doc2Vec.load('../models/Doc2Vec_AOL.model')
# --------------------------------------------------------------------

# Query auto-completion function
def query_completion(query, completion_list, session_queries, alpha=0.6):
    # Convert session queries to vectors
    session_queries_tokens = [word_tokenize(q) for q in session_queries]
    session_query_vectors = [model.infer_vector(q_tokens) for q_tokens in session_queries_tokens]

    # Frequency score (of each candidate query)
    combined_score = [frequencies.loc[a, 'Frequency'] if a in frequencies.index else 0 for a in completion_list]

    for i, candidate_query in enumerate(completion_list):  # Candidate queries
        candidate_score = 0  # similarity score of the candidate query

        # Convert candidate query to vector
        candidate_tokens = word_tokenize(candidate_query)
        candidate_query_vectors = model.infer_vector(candidate_tokens)

        # Compute similarity with all session queries
        for session_vector in session_query_vectors:
            candidate_score += model.wv.cosine_similarities(candidate_query_vectors, [session_vector])[0]

        combined_score[i] = alpha * candidate_score + (1 - alpha) * combined_score[i]

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
with open('../results/doc2vec_prefix5.txt', 'w') as f: f.write(str(mrr) + "\n" + "Sessions evaluated: " + str(num_sessions_evaluated))

prefix_length = 4  # query will be completed based on the first 4 characters
mrr, num_sessions_evaluated = evaluate_test_set_parallel(test, prefix_length, train, alpha=0.6)
print("MRR for prefix length", prefix_length, ":", mrr)
with open('../results/doc2vec_prefix4.txt', 'w') as f: f.write(str(mrr) + "\n" + "Sessions evaluated: " + str(num_sessions_evaluated))

prefix_length = 3  # query will be completed based on the first 3 characters
mrr, num_sessions_evaluated = evaluate_test_set_parallel(test, prefix_length, train, alpha=0.6)
print("MRR for prefix length", prefix_length, ":", mrr)
with open('../results/doc2vec_prefix3.txt', 'w') as f: f.write(str(mrr) + "\n" + "Sessions evaluated: " + str(num_sessions_evaluated))

prefix_length = 2  # query will be completed based on the first 2 characters
mrr, num_sessions_evaluated = evaluate_test_set_parallel(test, prefix_length, train, alpha=0.6)
print("MRR for prefix length", prefix_length, ":", mrr)
with open('../results/doc2vec_prefix2.txt', 'w') as f: f.write(str(mrr) + "\n" + "Sessions evaluated: " + str(num_sessions_evaluated))

prefix_length = 1  # query will be completed based on the first character
mrr, num_sessions_evaluated = evaluate_test_set_parallel(test, prefix_length, train, alpha=0.6)
print("MRR for prefix length", prefix_length, ":", mrr)
with open('../results/doc2vec_prefix1.txt', 'w') as f: f.write(str(mrr) + "\n" + "Sessions evaluated: " + str(num_sessions_evaluated))
