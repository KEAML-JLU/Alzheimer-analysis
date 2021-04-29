import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn import metrics
from sklearn import decomposition
import matplotlib.pyplot as plt
from itertools import cycle
from AP import takeSecond


def AP(word_all, word_lda_all):
    onehots = word_lda_all
    onehots = np.array(onehots)
    D = 1 - pdist(onehots, 'cosine')
    D1 = squareform(D, force="no", checks=True)
    Similarity = D1

    '''
     #similarity[0][0] = medium
    sum = np.sum(D)
    for i in range(size):
        Similarity[i][i] = sum/len(D) # sum = np.sum(D1[i])
    '''
    print("Similarity")

    preferenece = Similarity[Similarity!=0].min()
    # preferenece = np.median(Similarity)
    # for i in range(0,len(word_all)):
    #     Similarity[i][i]= 0
    #     # np.median(Similarity)
    print(preferenece)
    ap = AffinityPropagation(affinity='precomputed',max_iter=1000,damping=0.8,preference = preferenece).fit(Similarity)

    results = []
    words = word_all
    words = np.array(words)

    centers = []
    ida_spot=[]
    print("中心个数")
    print(len(ap.cluster_centers_indices_))
    for i in range(len(ap.cluster_centers_indices_)):
        centers.append(words[ap.cluster_centers_indices_[i]])
    for i in range(len(ap.cluster_centers_indices_)):
        ida_spot.append(word_lda_all[ap.cluster_centers_indices_[i]])
    for i in range(len(centers)):
        results.append([centers[i],ida_spot[i]])


    # labels = ap.labels_
    # for i in range(len(labels)):
    #     cluster = labels[i]
    #     center = centers[cluster]
    #     results.append([words[i], center])
    #     # 排序
    return results


if __name__ == '__main__':
    input_file = pd.read_csv('C:\\Users\\wenxj\\Desktop\\lda\\2013\\2013_final_790.txt',sep='\t')
    # print(type(input_file))
    data = input_file.values.tolist()
    # print(len(data))
    wordlist=[]
    for i in data:
        topic = i[0]
        wordlist.append(topic.split(','))

    word_lda_all = []
    word_all= []
    for topic in wordlist:
        word=[]
        word_lda = []
        for i in topic:
            word.append(i.split(':')[0])
            word_lda.append(float(i.split(':')[1])*100)
        word_all.append(word)
        word_lda_all.append(word_lda)

    result = AP(word_all, word_lda_all)

    with open('C:\\Users\\wenxj\\Desktop\\lda\\2010\\2010_center_test.csv', 'a', newline='', encoding='UTF-8') as f:
        for item in result:
            topic = item[0]
            ida_r = item[1]
            for j in  range(0,len(topic)):
                topic_csv = topic[j]+','+str(ida_r[j])
                f.write(topic_csv + '\n')
    结果一
    with open('C:\\Users\\wenxj\\Desktop\\lda\\2010\\2010_center_10.txt', 'a', newline='', encoding='UTF-8') as f:
        for i in result:
            center=list(i[1])
            topic = center[0]
            for j in range(1,len(center)):
                topic = topic+','+center[j]
            f.write(topic + '\n')


