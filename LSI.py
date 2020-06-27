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
import csv
import heapq

# document list
documents = []
with open("output_model_v1/doc_200_term_50_model/file-list",encoding="utf8") as file:
    while True:
        line = file.readline()
        if not line:
            break
        documents.append(line.replace("\n", ""))   

# vocab list
vocab = []
with open("output_model_v1/doc_200_term_50_model/vocab.all",encoding="utf8") as file:
    while True:
        line = file.readline()
        if not line:
            break
        vocab.append(line.replace("\n", ""))                     

# term count
with open("output_model_v1/doc_200_term_50_model/inverted-file") as file:
    term_count = 0
    while True:
        line = file.readline()
        if not line:
            break    
        voc_id1, voc_id2, count = line.split(" ")
        for i in range(int(count)):
            line = file.readline()
        term_count += 1     
term_count  

# doucument-term matrix
M = np.zeros([200,term_count])
with open("output_model_v1/doc_200_term_50_model/inverted-file") as file:
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
corpora_dict = []
for i in range(200):
    list = []
    list1 = []
    for j in range(term_count):
        if M[i][j] != 0:
            list.append((j,M[i][j])) 
            list1.append(dict_for_query[j])   
    list_all.append(list) 
    corpora_dict.append(list1)   

# 計算coherence衡量topic個數
dictionary1 = corpora.Dictionary(corpora_dict)
coherence_values = []
for number_of_topics in range(3, 7, 1):
    lsi=models.LsiModel(list_all, id2word=dictionary1, num_topics=number_of_topics)  
    coherencemodel = CoherenceModel(model=lsi, texts=corpora_dict, dictionary=dictionary1, coherence='c_v')
    coherence_values.append(coherencemodel.get_coherence())

# plot topics coherence
number_of_topics = range(3, 7, 1)
plt.plot(number_of_topics, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# train lsi model
lsi=models.LsiModel(list_all,id2word=dictionary,num_topics=5)
topics=lsi.show_topics(num_words=10,log=0)
for tpc in topics:
    print(tpc)


# 計算cos sim
cosine_similarity_matrix = similarities.MatrixSimilarity(lsi[list_all])  

# query test
def Main(number):
    # 將query-test的concepts切出
    xml = ET.parse('queries/query-test.xml')
    root = xml.getroot()
    topic = root.findall('topic')
    query = topic[number].find('concepts').text.replace("\n","").replace("。","")
    query = query.split("、")
    
    # 切出unigram和bigram
    test = []
    for i in range(len(query)):
        for j in range(len(query[i])):
            test.append(query[i][j])
            if j+1 < len(query[i]):
                test.append(query[i][j]+query[i][j+1])
    
    # 把存在query但不再training set的term拿掉 且 去掉重複字
    inter_dup = [a for a in test if a in dict_for_query]
    inter = []
    [inter.append(i) for i in inter_dup if not i in inter]

    #calculate LSI vector from word stem counts of the test document and the LSI model content
    query_test = []
    for i in range(len(inter)):
        query_test.append((dict_for_query.index(inter[i]),1))
    vector_lsi_test = lsi[query_test]
    # print("LSI Vector Test Document:", vector_lsi_test)

    #perform a similarity query against the corpus
    cosine_similarities_test = cosine_similarity_matrix[vector_lsi_test]
    # print("Cosine Similarities of Test Document LSI Vectors to Training Documents LSI Vectors:",cosine_similarities_test)

    # result
    # most_similar_document_test = documents[np.argmax(cosine_similarities_test)]
    top3 = heapq.nlargest(3, range(len(cosine_similarities_test)), cosine_similarities_test.take)
    most_similar_document_test = []
    for i in range(3):
        most_similar_document_test.append(documents[top3[i]])
    ranking_res = ""
    for i in range(len(most_similar_document_test)):
        ranking_res += most_similar_document_test[i][16:]+" " 
    return ranking_res    

if __name__  == "__main__":
    with open('LSI_ranking.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['query_id', 'retrieved_docs'])
        for i in range(10):
            top3 = Main(i)
            print('Top3 Documents for Query ', i ,':\n',top3) 
            writer.writerow([i, top3.lower()]) 