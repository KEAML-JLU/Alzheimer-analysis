# encoding:utf8
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim import models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import seaborn
import matplotlib.pyplot as plt
import numpy
import math

def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):
    """calculate the perplexity of a lda-model"""
    # dictionary : {7822:'deferment', 1841:'circuitry',19202:'fabianism'...]
    # print ('the info of this ldamodel: \n')
    # print ('num of testset: %s; size_dictionary: %s; num of topics: %s'%(len(testset), size_dictionary, num_topics))
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = [] # store the probablity of topic-word:[(u'business', 0.010020942661849608),(u'family', 0.0088027946271537413)...]
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    doc_topics_ist = [] #store the doc-topic tuples:[(0, 0.0006211180124223594),(1, 0.0006211180124223594),...]
    for doc in testset:
        doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0 # the probablity of the doc
        doc = testset[i]
        doc_word_num = 0 # the num of words in the doc
        for word_id, num in doc:
            prob_word = 0.0 # the probablity of the word
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic*prob_topic_word
            prob_doc += math.log(prob_word) # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum/testset_word_num) # perplexity = exp(-sum(p(d)/sum(Nd))
    # print ("the perplexity of this ldamodel is : %s"%prep)
    return prep






list_stopWords=list(set(stopwords.words('english')))
input_file_big = open("C:\\Users\\wenxj\\Desktop\\topic\\csv_year\\Ocean_2011.csv",'r',encoding='utf-8',errors='ignore').readlines()
# 转大小写
input_file = [text.lower() for text in input_file_big]
#分词
list_words=[word_tokenize(text) for text in input_file]
#过滤停止词
filtered_words=[[w for w in text if not w in list_stopWords ]for text in list_words]
# 过滤标点
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','’','≤','a.','b.','c.','d.','e.','m.', 'n.', 'p.', 'f.','g.','h.','i.','j.','k.','l.','o.','q.','r.','s.','t.','u.','v.','w.','x.','y.','z.']
text_list = [[word for word in text if word not in english_punctuations] for text in filtered_words ]
dropword = ['model','method','published','results','using','study','The','\'\'','``','two','paper','online']
text_list2 = [[word for word in text if word not in dropword] for text in text_list ]
#过滤数字
train_set = [[word for word in text if bool(re.search(r'\d', word))==False] for text in text_list2 ]
# res=[]
# for word in text_list2:
#     if bool(re.search(r'\d', word))==False:
#         res.append(word)
#     else:
#         pass
# 构建训练语料
dictionary = Dictionary(train_set)
dictionary.filter_extremes(no_below=40,no_above=0.1)
corpus_a = [dictionary.doc2bow(text) for text in train_set]
tfidf = models.TfidfModel(corpus_a)
corpus = tfidf[corpus_a]



#分训练、测试集
p = int(len(corpus) * .8)
cp_train = corpus[0:p]
cp_test = corpus[p:]
# lda模型训练
# 2013 	年开始50个主题
grid = dict()
for topic in range(200,500,10):
    # grid[topic]=[]
    grid[topic] = []
    lda = LdaModel(corpus=cp_train, id2word=dictionary, num_topics=topic,passes=2,update_every=0,alpha='auto',iterations = 500)
    # test_perplexity=lda.log_perplexity(corpus_a_test)
    # perplex= lda.bound(corpus_a_test)
    # test_perplexity = numpy.exp2(-perplex / sum(cnt for document in corpus_a_test for cnt in document))
    test_perplexity = perplexity(lda, cp_test, dictionary, len(dictionary.keys()), topic)
    print(topic)
    print(test_perplexity)
    grid[topic].append(test_perplexity)

df = pd.DataFrame(grid)

min_dot = numpy.argmin(df.iloc[0].values)
print(df.columns.values[min_dot],df.iloc[0].values[min_dot])
plt.figure(figsize=(20,8), dpi=120)
plt.subplot(221)
plt.plot(df.columns.values, df.iloc[0].values, '#007A99',linewidth=2)
plt.xticks(df.columns.values)
plt.ylabel('2011_test_perplexity')
plt.xlabel('number_of_topics')
plt.plot(df.columns.values[min_dot],df.iloc[0].values[min_dot],c=seaborn.xkcd_rgb['red'],marker='o')
plt.show()


#输出
lda = LdaModel(corpus=corpus_a, id2word=dictionary, num_topics=900,passes=2,update_every=0,alpha='auto',iterations = 500)
with open('C:\\Users\\wenxj\\Desktop\\lda\\2017_final_900.txt', 'a', newline='',encoding='UTF-8') as f:
    for i in range(0,900):
        input_str =lda.show_topic(i, topn=30)[0][0] + ':' + str(lda.show_topic(i, topn=30)[0][1])
        for j in range(1,len(lda.show_topic(i,topn=30))):
            word = lda.show_topic(i, topn=30)[j][0] + ':' + str(lda.show_topic(i, topn=30)[j][1])
            input_str = input_str +','+ word
        f.write(input_str+'\n')

