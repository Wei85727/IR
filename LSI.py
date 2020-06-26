import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import numpy as np
from gensim import models,corpora,similarities
import xml.etree.ElementTree as ET

# 將query-test的concepts切出
xml = ET.parse('queries/query-train.xml')
root = xml.getroot()
topic = root.findall('topic')
query = topic[3].find('concepts').text.replace("\n","").replace("。","")
query = query.split("、")

# 切出unigram和bigram
test = []
for i in range(len(query)):
    for j in range(len(query[i])):
        test.append(query[i][j])
        if j+1 < len(query[i]):
            test.append(query[i][j]+query[i][j+1])

# document list
documents = []
with open("output_model_v1/doc_2000_term_150_model/file-list",encoding="utf8") as file:
    while True:
        line = file.readline()
        if not line:
            break
        documents.append(line.replace("\n", ""))   

# vocab list
vocab = []
with open("output_model_v1/doc_2000_term_150_model/vocab.all",encoding="utf8") as file:
    while True:
        line = file.readline()
        if not line:
            break
        vocab.append(line.replace("\n", ""))                     

# doucument-term matrix
M = np.zeros([2000,300000])
with open("output_model_v1/doc_2000_term_150_model/inverted-file") as file:
    dictionary = {}
    dict_for_query = []
    a = 0
    while True:
        line = file.readline()
        if not line:
            break
        voc_id1, voc_id2, count = line.split(" ")
        dictionary[a] = vocab[int(voc_id1)]+vocab[int(voc_id2)]
        dict_for_query.append(vocab[int(voc_id1)]+vocab[int(voc_id2)])
        for i in range(int(count)):
            line = file.readline()
            docId, df, tfidf = line.split(" ")
            M[int(docId)][a] = float(tfidf)
        a += 1         

# 將matrix轉為套件input型式
list_all = []
for i in range(2000):
    list = []
    for j in range(300000):
        if M[i][j] != 0:
            list.append((j,M[i][j]))   
    list_all.append(list)    

# train lsi model
lsi=models.LsiModel(list_all,id2word=dictionary,num_topics=6)
topics=lsi.show_topics(num_words=10,log=0)
for tpc in topics:
    print(tpc)

# 計算cos sim
cosine_similarity_matrix = similarities.MatrixSimilarity(lsi[list_all])  

# 把存在query但不再training set的term拿掉
inter = [a for a in test if a in dict_for_query]

#calculate LSI vector from word stem counts of the test document and the LSI model content
query_test = []
for i in range(len(inter)):
    query_test.append((dict_for_query.index(inter[i]),1))
vector_lsi_test = lsi[query_test]
print("LSI Vector Test Document:", vector_lsi_test)

#perform a similarity query against the corpus
cosine_similarities_test = cosine_similarity_matrix[vector_lsi_test]
print("Cosine Similarities of Test Document LSI Vectors to Training Documents LSI Vectors:",
      cosine_similarities_test)

# 結果
most_similar_document_test = documents[np.argmax(cosine_similarities_test)]
print("Most similar Training Document to Test Document:", most_similar_document_test)     