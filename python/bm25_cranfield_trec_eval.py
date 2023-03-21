# BM25 model to retrieve documents for queries in the Cranfield dataset and output a model scores file in trec_eval format
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir("/content/drive/My Drive/CA6005_Mechanics_of_Search")
from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ETree
import pandas as pd
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.disabled = True # to disable logging

# Path to the Cranfield dataset
dataset_path = '/content/drive/My Drive/CA6005_Mechanics_of_Search/document'
# FIX: to parse the xml file for the document id and doc text, a fix is required to add a parent "newroot" to top and bottom of document XML file
docs_data_xml="/content/drive/My Drive/CA6005_Mechanics_of_Search/document/cran.all.1400.xml"

prstree = ETree.parse(docs_data_xml)
root = prstree.getroot()
doc_items = []
all_doc_items = []
for doc_data in root.iter('doc'):
    docid = doc_data.find('docno').text
    doctext = doc_data.find('text').text
  
    doc_items = [docid,
                 doctext]
    all_doc_items.append(doc_items)

# convert the list to a dictionary
all_doc_items_dict = dict(all_doc_items)

all_doc_items_df = pd.DataFrame(all_doc_items, columns=[
  'doc_id',
  'doc_text'])

doc_id_text_df = all_doc_items_df[['doc_id','doc_text']].copy()
# Load the queries from the dataset
queries_data_xml="/content/drive/My Drive/CA6005_Mechanics_of_Search/document/cran.qry.xml"
# parse the xml file for the query id and query text, to fix the query file, remove the last top in the file
prstree = ETree.parse(queries_data_xml)
root = prstree.getroot()
query_items = []
all_query_items = []
for query_data in root.iter('top'):
    queryid = query_data.find('num').text
    querytitle = query_data.find('title').text
  
    query_items = [queryid,querytitle]
    all_query_items.append(query_items)

# convert the list to a dictionary
all_query_items_dict = dict(all_query_items)

all_query_items_df = pd.DataFrame(all_query_items, columns=[
  'query_id','query_text'])
  
query_id_text_df = all_query_items_df[['query_id','query_text']].copy()

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('stopwords')
nltk.download('punkt')

def preprocess(text):
    # Tokenize the text
    words = nltk.word_tokenize(text.lower())

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Stem the words
    stemmer = SnowballStemmer('english')
    words = [stemmer.stem(word) for word in words]

    return words

#Indexing
#using Python's built-in defaultdict and Counter classes to create an inverted index
from collections import defaultdict, Counter

def build_index(docs):
    index = defaultdict(list)
    doc_lengths = {}

    for doc_id, doc in enumerate(docs):
        if doc is not None:
            terms = preprocess(doc)
            doc_lengths[doc_id] = len(terms)

            term_freqs = Counter(terms)
        else:
            logging.info('doc not processed as it is None is:'+str(doc))
        if doc is not None:
            for term, freq in term_freqs.items():
                index[term].append((doc_id, freq))
        else:
            logging.info('index not created for doc as its text is None is:'+str(doc))

    return index, doc_lengths
# To calculate the BM25 score, need to calculate the values below

# f: term frequency in the document
# qf: term frequency in the query
# df: document frequency of the term
# N: total number of documents in the collection
# dl: length of the document in terms
# avdl: average length of documents in the collection
# k1 and b: tuning parameters
#the following formula to calculate the BM25 score:

# score(d, q) = ∑ (idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avdl)) * (qf * (k2 + 1)) / (qf + k2))

import math

def bm25_score(query, index, doc_lengths, k1=1.2, b=0.75, k2=100):
    logging.info('query input to bm25_score function is:'+str(query))
    logging.info('index input to bm25_score function is:'+str(index))
    logging.info('doc_lengths input to bm25_score function is:'+str(doc_lengths))
    scores = defaultdict(float)
    query_terms = preprocess(query)
    logging.info('query_terms after preporcessing in the bm25_score function is:'+str(query_terms))

    N = len(doc_lengths)
    avdl = sum(doc_lengths.values()) / N

    for term in query_terms:
        if term not in index:
            continue

        df = len(index[term])
        logging.info('df is:'+str(df))
        logging.info('N is:'+str(N))
        idf = math.log((N - df + 0.5) / (df + 0.5))
        # Calculate the term frequency in the query
        qf = query_terms.count(term)
        logging.info('The term frequency (qf) in the query is:'+str(qf))
        
        for doc_id, freq in index[term]:
            f = freq
            dl = doc_lengths[doc_id]
            logging.info('idf in bm25_score function is:'+str(idf))
            logging.info('term in bm25_score function is:'+str(term))
            logging.info('doc_id in bm25_score function is:'+str(doc_id))
            logging.info('freq in bm25_score function is:'+str(freq))
            logging.info('doc_lengths in bm25_score function is:'+str(doc_lengths))
            logging.info('dl in bm25_score function is:'+str(dl))
            logging.info('doc_id in bm25_score function is:'+str(doc_id))
            score = idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avdl)) * (qf * (k2 + 1)) / (qf + k2)
            scores[doc_id] += score

    return scores
# format the output file as follows

# query_id Q0 doc_id rank score run_id

import os
import subprocess

index, doc_lengths = build_index(all_doc_items_dict.values())

results = {}
for query_id, query_text in all_query_items_dict.items():
    query_scores = bm25_score(query_text, index, doc_lengths)
    for doc_id, score in sorted(query_scores.items(), key=lambda x: x[1], reverse=True)[:100]:
        results.setdefault(query_id, []).append((doc_id, score))
        
os.chdir("/content/drive/My Drive/CA6005_Mechanics_of_Search/result")

with open('bm25_all_results.txt', 'w') as f:
    for query_id, doc_scores in results.items():
        for i, (doc_id, score) in enumerate(doc_scores):
            f.write(f'{query_id} Q0 {doc_id} {i+1} {score:.6f} BM25\n')
