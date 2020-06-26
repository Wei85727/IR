import os
import re
import jieba as ws
import pandas as pd
from gensim import models,corpora,similarities
import logging
import numpy as np
 
 
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
documents=[]
documents1=[]
labels=[]
stopwords = []
class_dir=os.listdir('dataset/')

# remove stopwords 
with open('stopwords.txt', 'r', encoding='UTF-8') as file:
    for data in file.readlines():
        data = data.strip()
        stopwords.append(data)

for i in class_dir:
    # if i.find('C')>-1:
    currentpath='dataset/'
#        print(currentpath)
    files=os.listdir(currentpath)
    for f in files:
        tmp_list=[]
        tmp_list1=[] # stopwords remainders  
        tmp_str=''
        try:            
            file=open(currentpath+f,encoding='utf8')
            file_str=file.read()
            file_str=re.sub('(\u3000)|(\x00)|(nbsp)','',file_str)
            doc=''.join(re.sub('[\d\W]','',file_str))
            tmp_str='|'.join(ws.cut(doc))
            tmp_list=tmp_str.split('|')    
            tmp_list1 = list(filter(lambda a: a not in stopwords and a != '\n', tmp_list))
            labels+=[i]
        except:
            print('read error: '+currentpath+f)
        documents1.append(file_str)    
        documents.append(tmp_list1) # stopwords removed
        # file.close()
    
tmp=[] 
test=[]
tmp_str=''
try:            
    file_str="半個月內發生多起搶劫案台北治安亮紅燈"
    file_str=re.sub('(\u3000)|(\x00)|(nbsp)','',file_str)
    doc=''.join(re.sub('[\d\W]','',file_str))
    tmp_str='|'.join(ws.cut(doc))
    tmp=tmp_str.split('|')
    test= list(filter(lambda a: a not in stopwords and a != '\n', tmp))
except:
    print('read error: '+currentpath+f)  
        
dictionary=corpora.Dictionary(documents)
corpus=[dictionary.doc2bow(doc) for doc in documents]#generate matrix
tf_idf=models.TfidfModel(corpus)

corpus_tfidf=tf_idf[corpus]
 
#training
lsi=models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=6)
topics=lsi.show_topics(num_words=10,log=0)
for tpc in topics:
    print(tpc)

#calculate cosine similarity matrix for all training document LSI vectors
cosine_similarity_matrix = similarities.MatrixSimilarity(lsi[corpus])
# print("Cosine Similarities of LSI Vectors of Training Documents:",
#       [row for row in cosine_similarity_matrix])

#calculate LSI vector from word stem counts of the test document and the LSI model content
vector_lsi_test = lsi[dictionary.doc2bow(test)]
print("LSI Vector Test Document:", vector_lsi_test)

#perform a similarity query against the corpus
cosine_similarities_test = cosine_similarity_matrix[vector_lsi_test]
print("Cosine Similarities of Test Document LSI Vectors to Training Documents LSI Vectors:",
      cosine_similarities_test)

#get text of test documents most similar training document
most_similar_document_test = documents1[np.argmax(cosine_similarities_test)]
print("Most similar Training Document to Test Document:", most_similar_document_test)


# test string

# 台灣與中共海峽兩岸關係的問題幾十年來還有待解決 chi
# 金融海嘯造成全球股市崩盤台灣跟著受到影響 eco
# 教育部宣布從今天起國小學生禮拜三只能上半天課 edu
# 雲林茶農研發出新品種專家讚不絕口 loc
# 半個月內發生多起搶劫案台北治安亮紅燈 soc
# 在九局下半最後一個打席敲出再見安打逆轉戰局 spo



