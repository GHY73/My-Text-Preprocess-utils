# -*- coding: utf-8 -*-
# @Time :   19-5-5 下午4:06 
# @Author :     HaiYang Gao
# @OS：  ubuntu 16.04
# @File :   text_proprecess.py 

from collections import Counter
import tensorflow.contrib.keras as kr
import jieba

# 打开文件
def open_file(filepath, mode='r'):
    return open(filepath, mode, encoding='utf-8', errors='ignore')


# 读取文件数据
def read_file(filepath):
    traindatas, labels = [], []
    with open_file(filepath) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    traindatas.append(content)
                    labels.append(label)
            except:
                print('step1：请检查文件路径是否正确')
                print('step2：请检查文件分割符是否为空格')
    return traindatas, labels


# 构建词汇表，存储
def build_vocab(traindatas, vocab_dir, vocab_size=5000):
    all_data = []
    for content in traindatas:
        all_data.extend(content)
    counter = Counter(all_data)
    # 存在数量一样的word，因此每次排序的时候，不稳定，导致构建的词汇表不完全一致
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words))


# 读取词汇表，构造word2id字典{'word':id以及id2word字典{id:'word'}
def make_map_dict(vocab_dir):
    with open_file(vocab_dir) as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict(zip(range(len(words)), words))
    return words, word_to_id, id_to_word


# file2id，返回file的id表示
def file2id(traindatas,max_length=3000):
    _, word2id, id2word = make_map_dict('vocab.txt')
    data_id = []
    for i in range(len(traindatas)):
        data_id.append([word2id[x] for x in traindatas[i] if x in word2id])
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    traindatas_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)

    return traindatas_pad


# id2file，返回原始文本
def id2file(id_content, id2word):
    return ''.join(id2word[x] for x in id_content if x > 0)

# jieba分词
def wordsegment(traindatas):
    cut_data=[]
    for each in traindatas:
        a = ' '.join(jieba.cut(each,cut_all=False))
        cut_data.append(a)
    return cut_data


"""
下面是一个简单的调用过程例子
"""
if __name__== '__main__':
    # step1：读文件
    traindatas, labels = read_file('cnews.test.txt')
    # step2：为训练数据建词汇表，直接写入本地
    build_vocab(traindatas,'vocab.txt')
    # step3：按照词汇表，生成word-id的映射
    _, word2id, id2word = make_map_dict('vocab.txt')
    # step4：训练数据转为id表示，返回narray
    datas_id = file2id(traindatas,max_length=3000)
    # step5:id转为训练数据，每次转一条数据，多行数据，需要循环调用
    datas_word = id2file(datas_id[0],id2word)

    """
    一般预处理，第一步是分词，之后再进行step1-step4。
    这里只简单调用一下。
    如果需要先分词，自己调整处理步骤或逻辑。
    """
    cut_data = wordsegment(traindatas)
    print('=============================Run over=============================')
