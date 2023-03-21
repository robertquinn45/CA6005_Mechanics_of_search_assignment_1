# Vector Space Model to retrieve documents for queries in the Cranfield dataset and output a model scores file in trec_eval format
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir("/content/drive/My Drive/CA6005_Mechanics_of_Search")
from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ETree
import pandas as pd
import logging

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

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
##################################################
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

#####################################
# Preprocess the documents and queries by tokenizing, removing stop words, and stemming the words
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Preprocess the documents and queries
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and token.isalpha()]
    return tokens

processed_documents = {}
for doc_id, doc_text in all_doc_items_dict.items():
    if doc_text is not None:
        processed_documents[doc_id] = preprocess(doc_text)
    else:
      logging.info('doc_text not processed as it is None is:'+str(doc_text)) 

logging.info('processed_documents is:'+str(processed_documents))

processed_queries = {}
for query_id, query_text in all_query_items_dict.items():
    processed_queries[query_id] = preprocess(query_text)

logging.info('processed_queries is:'+str(processed_queries))

# Create a dictionary of all the unique terms in the documents and queries, and assign an index to each term
term_dict = {}
for doc_id, doc_tokens in processed_documents.items():
    for token in doc_tokens:
        if token not in term_dict:
            term_dict[token] = len(term_dict)

for query_id, query_tokens in processed_queries.items():
    for token in query_tokens:
        if token not in term_dict:
            term_dict[token] = len(term_dict)

logging.info('term_dict is:'+str(term_dict))
# term_dict is all the unqiue words listed once across each docmuent, with an index number beside it
# Create a term-document matrix, where each row represents a term and each column represents a document, and the value in each cell represents the frequency of the term in the document. You can use the following code to create the matrix:

import numpy as np

# Create a term-document matrix - the term_doc_matrix calcs to have terms in the columns and documents on the rows, the cosine similarity computing in a section further down the script, 
#   expects the columns to be the same number for both matrices, hence the term_doc_matrix has terms in the columns and documents on the rows as code below
term_doc_matrix = np.zeros((len(all_doc_items_dict),len(term_dict)))
for doc_id, doc_tokens in processed_documents.items():
    for token in doc_tokens:
        term_doc_matrix[int(doc_id) - 1, term_dict[token]] += 1

logging.info('term_doc_matrix is:'+str(term_doc_matrix))
#Create a query-term matrix, each row represents a query and each column represents a term
# use the following code to create the query-term matrix:
# Create a query-term matrix
logging.info('len(all_query_items_dict) is:'+str(len(all_query_items_dict)))
logging.info('len(term_dict) is:'+str(len(term_dict)))
# Find the maximum query id number in the query list and use this to size the amount of rows in the query_term_matrix array
# Extract the column "queryid" as a list from the dictionary "all_query_items_dict"
logging.info('type(all_query_items_dict) is:'+str(type(all_query_items_dict)))
logging.info('np.shape(all_query_items_dict) is:'+str(np.shape(all_query_items_dict)))
all_query_items_column_names = list(all_query_items_dict.keys())
logging.info('columns in all_query_items_dict are:'+str(all_query_items_column_names))
all_query_items_column_names_trimmed_list = [int(x.strip()) for x in all_query_items_column_names]
all_query_items_max_value = max(all_query_items_column_names_trimmed_list)
logging.info('all_query_items_max_value is:'+str(all_query_items_max_value))

query_term_matrix = np.zeros((int(all_query_items_max_value), len(term_dict))) #numpy.zeros(creates a matrix of zeros with input parameters (rows, columns)

for query_id, query_tokens in processed_queries.items():
    for token in query_tokens:
        query_term_matrix[int(query_id) - 1, term_dict[token]] += 1

logging.info('query_term_matrix is:'+str(query_term_matrix))

#Compute the TF-IDF weights for the term-document matrix

from sklearn.feature_extraction.text import TfidfTransformer

# Compute the TF-IDF weights
# norm: The normalization scheme to use for the TF-IDF weights. 
# 'l2' normalization scales the weights so that the sum of the squares of each row is 1
# smooth_idf: a smoothing factor to the IDF (inverse document frequency) weights to prevent division by zero. If True, a smoothing factor of 1 is added to the IDF weights, otherwise no smoothing is applied
# use_idf: use IDF weighting in addition to TF (term frequency) weighting. If True, the TF-IDF weights are computed as tf * idf, whereas tf is the raw term frequency in the document and idf is the inverse document frequency of the term.
tfidf = TfidfTransformer(norm='l2', smooth_idf=True, use_idf=True)  
tfidf.fit(term_doc_matrix)
tfidf_weights = tfidf.transform(term_doc_matrix).toarray()
logging.info('tfidf_weights is:'+str(tfidf_weights))
# Compute the cosine similarity between each query and document using the query-term matrix and the TF-IDF weights
from sklearn.metrics.pairwise import cosine_similarity

# Compute the cosine similarity
similarity = cosine_similarity(query_term_matrix, tfidf_weights)
logging.info('similarity is:'+str(similarity))
# Save the results in a format that can be evaluated using trec_eval
os.chdir("/content/drive/My Drive/CA6005_Mechanics_of_Search/result")
with open('vsm_similarity_results.txt', 'w') as f:
    for i, query_id in enumerate(all_query_items_dict.keys()):
        sim = similarity[i]
        sorted_indices = sim.argsort()[::-1]
        for j in sorted_indices:
            doc_id = str(j + 1)
            score = sim[j]
            f.write(f'{query_id} Q0 {doc_id} 0 {score} vector_space_model\n')

# Attempt below at getting trec_eval installed and working on Google Colab. It did not work, so installed the trec_eval locally on the Windows laptop.
# Copied the VSM results from file above (vsm_similarity_results.txt) and the cranqrel.trec.txt to the local Windows directpry to generate the trec_eval output

# Evaluate the results using trec_eval
# try with the smaller query 1,2 and doc 1,2 for the qrels file and see what happens
#os.chdir("/content/drive/My Drive/CA6005_Mechanics_of_Search/trec_eval")
#!sudo apt-get install build-essential
#!wget https://github.com/usnistgov/trec_eval/archive/master.zip
#!unzip master.zip
#os.chdir("/content/drive/My Drive/CA6005_Mechanics_of_Search/trec_eval/trec_eval-master")
#!cd trec_eval-master
#!pwd
#!make
#!sudo make install

#!sudo add-apt-repository universe
#!sudo apt-get update
#!sudo apt-get install trec_eval
#logging.info('after step sudo apt-get install trec_eval')
#!pwd
#!which trec_eval
#os.chdir("/content/drive/My Drive/CA6005_Mechanics_of_Search/trec_eval/trec_eval-master")
#import subprocess

# Set the path to the trec_eval executable
#trec_eval_path = '/content/drive/My Drive/CA6005_Mechanics_of_Search/trec_eval/trec_eval-master'

# Set the path to the relevance judgments file
#qrels_path = '/content/drive/My Drive/CA6005_Mechanics_of_Search/relevance'

# Set the path to the search results file
#results_path = '/content/drive/My Drive/CA6005_Mechanics_of_Search/result'

# Call trec_eval and capture the output
#output = subprocess.check_output([trec_eval_path, '-q', qrels_path, results_path])

# Print the output
#print(output.decode('utf-8'))
