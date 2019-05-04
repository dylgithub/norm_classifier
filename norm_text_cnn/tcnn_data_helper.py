#encoding=utf-8
import pandas as pd
import numpy as np
import jieba
import jieba.posseg as pseg
from gensim.models import word2vec
from collections import Counter
import tensorflow.contrib.keras as kr
#把语料进行切词之后再保存
#因为若语料过大，每次切词很浪费时间
def get_cut_data(file_location,save_location,use_dict,dict_location=None,encoding='utf-8'):
    '''
    :param file_location: 读取的文本的地址
    :param save_location: 保存切词后文本以及对应词性文件的地址
    :param use_dict: 是否使用自定义词典
    :param dict_location: 若使用自定义传入自定义词典的路径
    :param encoding: 文件的编码格式，默认为utf-8
    :return:
    '''
    #注意这里默认读取和保存的文件都是.csv的，其中content代表文本列,label代表文本相对应的类别
    #这里使用的分词工具是jieba
    df = pd.read_csv(file_location,encoding=encoding)
    #获得语料文本
    content_list=list(df['content'])
    label_list=list(df['label'])
    if use_dict:
        jieba.load_userdict(dict_location)
    #获得分词后的词以及词性的结果，以'_sp_'隔开
    cut_content_list=[]
    cut_pseg_list=[]
    for content in content_list:
        word_list=[]
        pseg_list=[]
        for w in pseg.cut(content):
            word_list.append(w.word)
            if w.flag[0] in ['v','n','l','y','r']:
                pseg_list.append(w.flag[0])
            else:
                pseg_list.append('x')
        cut_content_list.append('_sp_'.join(word_list))
        cut_pseg_list.append('_sp_'.join(pseg_list))
    # cut_content_list=['_sp_'.join(jieba.lcut(content)) for content in content_list]
    save_df=pd.DataFrame({'content':cut_content_list,'pseg':cut_pseg_list,'label':label_list})
    save_df.to_csv(save_location,encoding=encoding,index=False)
#获得训练数据以及其对应的类别标签
def get_data_label(file_location,encoding='utf-8'):
    '''
    :param file_location: 读取的文件地址，这里默认需要的文件是上一步生成的.csv文件
    :param encoding: 文件默认的编码
    :return: content_list一个List其中的元素都是list,每个元素list代表一条文本，list中每个元素是一个str是切词后的一个词
              label_list每条训练文本对应的类别标签List
    '''
    df = pd.read_csv(file_location,encoding=encoding)
    cut_content_list=list(df['content'])[:3000]
    cut_pseg_list=list(df['pseg'])[:3000]
    label_list=list(df['label'])
    #有些文本会以空格作为间隔，因此这里使用_sp_作为分隔标记没使用空格
    content_list=[cut_content.split('_sp_') for cut_content in cut_content_list]
    pseg_list=[cut_pseg.split('_sp_') for cut_pseg in cut_pseg_list]
    return content_list,pseg_list,label_list[:3000]
def train_w2v_model(file_location,model_save_location,encoding='utf-8'):
    content_list,_,_=get_data_label(file_location,encoding)
    w2v_model=word2vec.Word2Vec(content_list,min_count=1,size=80)
    w2v_model.save(model_save_location)
def get_label2id_dict():
    label2id_dict={'教育':0,'时尚':1, '家居':2, '娱乐':3, '财经':4, '体育':5, '房产':6, '游戏':7, '时政':8, '科技':9}
    return label2id_dict
#根据训练语料构建词汇表以及词性表进行存储
def build_vocab(file_location,vocab_location,pseg_location,top_n=None,encoding='utf-8'):
    content_list,_,_=get_data_label(file_location,encoding)
    all_words = []
    for content in content_list:
        all_words.extend(content)
    words=None
    #若所有的词都存入词汇表
    #额外添加'<PAD>'以及'UNKNOW'是为了padding 0以及padding不存在于词汇表中的词做准备
    if not top_n:
        all_words=list(set(all_words))
        words=['<PAD>']+all_words+['UNKNOW']
    #若只选取top_n放入词汇表
    else:
        counter = Counter(all_words)
        count_pairs = counter.most_common(min(len(all_words),top_n))
        temp_words,_ =zip(*count_pairs)
        temp_words=list(temp_words)
        words = ['<PAD>'] + temp_words + ['UNKNOW']
    with open(vocab_location,'w',encoding=encoding) as fw:
        for word in words:
            fw.write(word+'\n')
    pseg_list=['<PAD>'] + ['v','n','l','y','r','x'] + ['UNKNOW']
    with open(pseg_location,'w',encoding=encoding) as fw:
        for pseg in pseg_list:
            fw.write(pseg+'\n')
#获得训练语料的词表词性表以及词向id词性向id映射的字典
def get_word_and_pseg2id_dict(vocab_location,pseg_location,encoding='utf-8'):
    words=[]
    psegs=[]
    with open(vocab_location, 'r', encoding=encoding) as fr:
        for line in fr.readlines():
            new_line=line.strip('\n')
            words.append(new_line)
    with open(pseg_location, 'r', encoding=encoding) as fr:
        for line in fr.readlines():
            new_line=line.strip('\n')
            psegs.append(new_line)
    word2id_dict = dict(zip(words,range(len(words))))
    pseg2id_dict = dict(zip(psegs,range(len(psegs))))
    return words,word2id_dict,psegs,pseg2id_dict
#对训练数据进行处理，返回词表中各词对应的词向量,padding后语料向id的转换以及one_hot的label
def process_file(file_location,w2v_model_location,vocab_location,psegs_location,label_is_id,sen_length,vec_size,pseg_size,encoding='utf-8'):
    content_list,pseg_list,label_list=get_data_label(file_location,encoding)
    word2vec_model=word2vec.Word2Vec.load(w2v_model_location)
    words,word2id_dict,psegs,pseg2id_dict=get_word_and_pseg2id_dict(vocab_location,psegs_location)
    word_vecs=[]
    for i,word in enumerate(words):
        if i==0:
            word_vecs.append([0]*vec_size)
        else:
            if word in word2vec_model.wv.vocab.keys():
                word_vecs.append(list(word2vec_model[word]))
            else:
                word_vecs.append(list(np.random.uniform(-0.25,0.25,vec_size)))
    #获得了词表中的词向量
    word_vecs = np.array(word_vecs).astype(np.float32)
    psegs_vec=[]
    for i,pseg in enumerate(psegs):
        if i==0:
            psegs_vec.append([0]*pseg_size)
        else:
            psegs_vec.append(list(np.random.uniform(-0.25,0.25,pseg_size)))
    #获得词性向量，这里是随机生成的
    psegs_vec = np.array(psegs_vec).astype(np.float32)
    #把文本中词向id转换
    sen_index = []
    for sen in content_list:
        words_index = []
        for word in sen:
            if word in word2id_dict:
                words_index.append(word2id_dict[word])
            #若只选取top_n加入词表，则会存在词表中不存在的词都把其id设置为UNKNOW
            else:
                words_index.append(word2id_dict['UNKNOW'])
        sen_index.append(words_index)
    # 这里进行的是padding操作，根据设定的句子的最大长度，长的截取，短的补0
    # 注意：1.sen_index中把词转换为相应的id，2.默认的为从前面填充0,3.默认的长度过长的从前面截取
    sen_index = kr.preprocessing.sequence.pad_sequences(sen_index, sen_length)
    # 把文本中词性向id转换
    sen_pseg_index=[]
    for psegs in pseg_list:
        psegs_index=[]
        for pseg in psegs:
            if pseg in pseg2id_dict:
                psegs_index.append(pseg2id_dict[pseg])
            else:
                psegs_index.append(pseg2id_dict['UNKNOW'])
        sen_pseg_index.append(psegs_index)
    sen_pseg_index = kr.preprocessing.sequence.pad_sequences(sen_pseg_index, sen_length)
    #若类别标签不是数字形式的应先转换
    if not label_is_id:
        label2id_dict=get_label2id_dict()
        label_list=[label2id_dict[i] for i in label_list]
    #把id形式的类别标签转换为one_hot形式的
    one_hot_label=kr.utils.to_categorical(label_list, num_classes=len(set(label_list)))
    return word_vecs,sen_index,psegs_vec,sen_pseg_index,one_hot_label
#生成批次数据
def batch_iter(x,x_pseg,y,batch_size=64):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    #打乱数据，若数据已打乱则不需要
    # indices = np.random.permutation(np.arange(data_len))
    # x_shuffle = x[indices]
    # y_shuffle = y[indices]
    #每次返回一个批次的词、词性以及类别标签
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size,data_len)
        yield x[start_id:end_id],x_pseg[start_id:end_id],y[start_id:end_id]
#生成不包含词性的批次数据
def batch_iter2(x,y,batch_size=64):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    #打乱数据，若数据已打乱则不需要
    # indices = np.random.permutation(np.arange(data_len))
    # x_shuffle = x[indices]
    # y_shuffle = y[indices]
    #每次返回一个批次的词、词性以及类别标签
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size,data_len)
        yield x[start_id:end_id],y[start_id:end_id]
if __name__ == '__main__':
    get_cut_data('data/train_data.csv','data/cut_pseg_train_data.csv',False)
    # content_list,_=get_data_label('data/cut_train_data.csv')
    # print(len(content_list))
    # train_w2v_model('data/cut_train_data.csv','w2v_model/w2v_model')
    # build_vocab('data/cut_train_data.csv','words/words.txt')
    # words, word2id_dict=get_word2id_dict('words/words.txt')
    # print(words[:5])
    # print(word2id_dict)
    # word_vecs, sen_index,one_hot_label=process_file('data/cut_train_data.csv','w2v_model/w2v_model','words/words.txt',False,200,80)
    # print(word_vecs.shape)
    # print(sen_index.shape)
    # print(one_hot_label.shape)
    # label2id_dict()