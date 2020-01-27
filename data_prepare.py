# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:30:58 2020

@author: us
"""
import jieba
import numpy as np
import pandas as pd
import pickle
import re
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

data = pd.read_table('./kuakua_500.txt',header=None)

# 整理夸夸语料，把求夸内容和夸夸内容，整理成dict
kuakua_dict = {}
for title in data[0][:501]:
    print(eval(title)["title"])
    kuakua_dict[eval(title)["title"]] = []
    for content in eval(title)["replies"]:
        print(content['content'])
        kuakua_dict[eval(title)["title"]].append(content['content'])


#把dict存入文件，需要的时候直接读取，不临时计算
fileHandle = open('kuakua_corp_dict.file', 'wb')  
pickle.dump(kuakua_dict, fileHandle) 
fileHandle.close() 

#取出dict
fileHandle = open('kuakua_corp_dict.file', 'rb')
word_v = pickle.load(fileHandle)
fileHandle.close()


# 测试

#把dict中title转为句子向量，后期 按向量匹配话题相似度
class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector

class Sentence:
    def __init__(self, word_list):
        self.word_list = word_list

    def len(self) -> int:
        return len(self.word_list)

def get_frequency_dict(file_path):
    fileHandle = open(file_path, 'rb')
    freq_dict = pickle.load(fileHandle)
    fileHandle.close()
    return freq_dict

def get_word_frequency(word_text, freq_dict):
    if word_text in freq_dict:
        freq = freq_dict[word_text]
        # print(freq)
        return freq
    else:
        return 1.0

def get_word2vec(file_path):
    # print('正在载入词向量...')
    fileHandle = open(file_path, 'rb')
    word_v = pickle.load(fileHandle)
    fileHandle.close()
    return word_v

# sentence_to_vec方法就是将句子转换成对应向量的核心方法
def sentence_to_vec(model_v, allsent, freq_dict, embedding_size: int, a: float = 1e-3):

    sentence_set = []
    for sentence in allsent:
        vs = np.zeros(embedding_size)
        # add all word2vec values into one vector for the sentence
        sentence_length = sentence.len()
        # print(sentence.len())
        # 这个就是初步的句子向量的计算方法
        #################################################
        for word in sentence.word_list:
            # print(word.text)
            a_value = a / (a + get_word_frequency(word.text, freq_dict))
            # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, word.vector))
            # vs += sif * word_vector

        vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)
        # add to our existing re-calculated set of sentences
    #################################################
    # calculate PCA of this sentence set,计算主成分
    pca = PCA()
    # 使用PCA方法进行训练
    pca.fit(np.array(sentence_set))
    # 返回具有最大方差的的成分的第一个,也就是最大主成分,
    # components_也就是特征个数/主成分个数,最大的一个特征值
    u = pca.components_[0]  # the PCA vector
    # 构建投射矩阵
    u = np.multiply(u, np.transpose(u))  # u x uT
    # judge the vector need padding by wheather the number of sentences less than embeddings_size
    # 判断是否需要填充矩阵,按列填充
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            # 列相加
            u = np.append(u, 0)  # add needed extension for multiplication below

    # resulting sentence vectors, vs = vs -u x uT x vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u, vs)
        sentence_vecs.append(np.subtract(vs, sub))
    return sentence_vecs

def get_sentence(model_v, train='分化凸显领军自主 电动智能贯穿汽车变革'):

    allsent = []
    for each in train:
        sent1 = list(jieba.cut(each, cut_all=False))
        print(sent1)
        s1 = []
        for word in sent1:
            print(word)
            try:
                vec = model_v[word]
            except KeyError:
                vec = np.zeros(100)
            s1.append(Word(word, vec))
        ss1 = Sentence(s1)
        allsent.append(ss1)
    return allsent

# 获取已训练好的词向量结果
model_v = get_word2vec('./word2vec_File.file')


input_news = ['27岁赚到100万']



# 把dict中key都转为vector#############################################
kuakua_vec_dict = {}

kuakua_dict.keys()


for t in kuakua_dict.keys():
    
    v_c = get_sentence(model_v, [t])

    def SIF(v_c):
        # v_t,content_v,v_c 按照SIF模型向量化
        # print('正在载入词频...')
        freq_dict = get_frequency_dict('./word2vec_File.file')
    
        v_c = sentence_to_vec(model_v, v_c, freq_dict, 100, 1e-3)
        return v_c
    
    v_c = SIF(v_c)
    
    kuakua_vec_dict[str(v_c)] = kuakua_dict[t]
    print(t)


    

#把dict存入文件，需要的时候直接读取，不临时计算
fileHandle = open('kuakua_vec_dict.file', 'wb')  
pickle.dump(kuakua_vec_dict, fileHandle) 
fileHandle.close() 

#取出dict
fileHandle = open('kuakua_vec_dict.file', 'rb')
word_v = pickle.load(fileHandle)
fileHandle.close()




