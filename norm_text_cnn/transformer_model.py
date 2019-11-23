#encoding=utf-8
import tensorflow as tf
import tcnn_data_helper
from Multi_Head_self_attention_model import positional_encoding
import transformer_encoder
#用于产生权重向量
def W_generate(shape):
    inital=tf.truncated_normal(shape)
    return tf.Variable(inital)
#用于产生偏置值
def bias_generate(shape):
    inital=tf.constant(0.1,shape=[shape])
    return tf.Variable(inital)
'''
该类用于完成cnn模型的正向传播，返回一个正向传播的结果
sentence_length：每个句子的长度
vector_size：每个单词向量的维度
num_classes：分类的类别数
num_filters：卷积核的个数
filter_hs：所使用的各个高度的卷积核
'''
#主要是记录transformer的位置编码如何添加
class TransformerModel():
    def __init__(self,config):
        self.config = config
        # 四个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.sentence_length], name='input_x')
        self.input_x_pseg = tf.placeholder(tf.int32, [None, self.config.sentence_length], name='input_x_pseg')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.transformer()
    #建立模型，执行正向传播，返回正向传播得到的值
    def transformer(self):
        # 向量映射
        word_vecs, sen_index, psegs_vec, sen_pseg_index, one_hot_label = tcnn_data_helper.process_file(
            self.config.file_location, self.config.w2v_model_location, self.config.words_location, self.config.psegs_location,
            False, self.config.sentence_length, self.config.vector_size, self.config.pseg_size)
        with tf.name_scope('embedding'):
            input_x_content = tf.nn.embedding_lookup(word_vecs,self.input_x)
            # print(tf.shape(self.input_x)[0])
            #获得位置编码的部分
            with tf.variable_scope("positional_encoding"):
                #返回的是维度为[self.config.sentence_length,self.config.vector_size]的tensor
                positional_embedding=positional_encoding(self.config.vector_size,self.config.sentence_length)
                #注意此处的维度不能固定死为self.config.batch_size，因为最后一批次的训练数据大小往往不为batch_size除非是训练数据的大小正好能被batch_size整除
                #应该保持和填充的数据self.input_x的0维相同（即为填充数据其批次的大小）
                #注意应该用tf.shape(self.input_x)[0]获得而不可用self.input_x.shape[0]获得
            positional_inputs=tf.tile(tf.range(0,self.config.sentence_length),[tf.shape(self.input_x)[0]])
            #返回batch_size个x，每个x没[0,1,2....self.config.sentence_length]
            positional_inputs=tf.reshape(positional_inputs,[tf.shape(self.input_x)[0],self.config.sentence_length])
            #融入位置信息
            input_x_content_add_positional=tf.add(input_x_content,tf.nn.embedding_lookup(positional_embedding,positional_inputs))
            # print("shape test:",input_x_content.shape)
            input_x_pseg = tf.nn.embedding_lookup(psegs_vec,self.input_x_pseg)
            _input_x = tf.concat((input_x_content_add_positional,input_x_pseg), axis=-1)
        #随后通过transformer提取特征
        outputs=transformer_encoder.Encoder().build(_input_x)
        #全连接层操作
        print('begin full_connection')
        with tf.name_scope("full_connection"):
            #维度转换之后取最后一时刻的输出
            outputs = tf.transpose(outputs, [1, 0, 2])
            f_input=outputs[-1]
            #分类器
            W = tf.Variable(tf.truncated_normal([self.config.vector_size+self.config.pseg_size, self.config.num_classes]))
            self.l2_loss = tf.nn.l2_loss(W)
            b = tf.Variable(tf.constant(0., shape=[self.config.num_classes]), name="b")
            self.y_pred = tf.nn.xw_plus_b(f_input, W, b, name="scores")  # wx+b
            #预测类别
            self.pred_label = tf.argmax(tf.nn.softmax(self.y_pred), 1)
        with tf.name_scope('optimize'):
            # 计算损失值,会自动计算softmax故必须传入没计算之前的，否则相当于计算了两次
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y,logits=self.y_pred)
            self.losses = tf.reduce_mean(cross_entropy)
            # 加L2正则化
            # self.losses = tf.reduce_mean(cross_entropy) + self.config.l2_reg_lambda *self.l2_loss
            #优化器
            self.optim = tf.train.AdamOptimizer(self.config.init_learning_rate).minimize(self.losses)
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.pred_label, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
