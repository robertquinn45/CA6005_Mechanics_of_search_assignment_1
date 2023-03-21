# multinomial language model to retrieve documents for queries in the Cranfield dataset and output a model scores file in trec_eval format
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir("/content/drive/My Drive/CA6005_Mechanics_of_Search")
from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ETree
import pandas as pd
import logging
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.disabled = True # to disable logging

# Path to the Cranfield dataset
dataset_path = '/content/drive/My Drive/CA6005_Mechanics_of_Search/document'
# FIX: to parse the xml file for the document id and doc text, a fix is required to add a parent "newroot" to top and bottom of document XML file
docs_data_xml="/content/drive/My Drive/CA6005_Mechanics_of_Search/document/cran.all.1400.xml" #docs_small_correct.xml

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
# convert the list to a Series
#all_doc_items_series = pd.Series(all_doc_items)

all_doc_items_df = pd.DataFrame(all_doc_items, columns=[
  'doc_id',
  'doc_text'])
  
#print(xmlToDf.to_string(index=False))
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
  
#print(xmlToDf.to_string(index=False))
query_id_text_df = all_query_items_df[['query_id','query_text']].copy()
#####################################
# Preprocess via tokenizing, removing stop words, and stemming the words. 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Preprocess the documents and queries
stop_words = set(stopwords.words('english'))


corpus = [] 
for doc_id, doc_text in all_doc_items_dict.items():
    logging.info('doc_id before processing is :'+str(doc_id))
    if doc_text is not None:
        doc_text_lower=doc_text.lower()
        logging.info('doc_text_lower before processing is :'+str(doc_text_lower))
        corpus.append(' '.join([word for word in doc_text_lower.split() if word not in stop_words]))
    else:
      logging.info('doc_text not processed as it is None is:'+str(doc_text)) 

################################################
#Multinomial language specifc model section

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

logging.info('type(corpus) is:'+str(type(corpus)))
logging.info('np.shape(corpus) is:'+str(np.shape(corpus)))
logging.info('corpus is:'+str(corpus))
# Build the model
vectorizer = CountVectorizer()
logging.info('vectorizer is:'+str(vectorizer))
X = vectorizer.fit_transform(corpus)
logging.info('X is:'+str(X))
y = range(len(corpus))
logging.info('y is:'+str(y))
clf = MultinomialNB(alpha=1.0)
logging.info('clf is:'+str(clf))
clf.fit(X, y)
logging.info('clf after command clf.fit(X, y) is:'+str(clf))

results = []
for query_id, query_text in all_query_items_dict.items():
    # pass into vectorizer.transform in the following format: 0 What day is it?
    #                                                         1 When is Patricks day?
    logging.info('query_id is:'+str(query_id))
    logging.info('query_text is:'+str(query_text))
    query_vector = vectorizer.transform([query_text])
    logging.info('query_vector is:'+str(query_vector))
    scores = clf.predict_log_proba(query_vector)
    logging.info('type(scores) is:'+str(type(scores)))
    logging.info('np.shape(scores) is:'+str(np.shape(scores)))
    logging.info('scores is:'+str(scores))
    for j, score in enumerate(scores):
        logging.info('j is:'+str(j))
        logging.info('score is:'+str(score))
        results.append((str(query_id), "Q0", str(int(j)+1), str(int(j)),str(score[1]), "MMLM"))
os.chdir("/content/drive/My Drive/CA6005_Mechanics_of_Search/result")
with open('multinomial_language_results_all.txt', 'w') as f:
    for result in results:
        logging.info('result is:'+str(result))
        f.write(" ".join(result) + "\n")
