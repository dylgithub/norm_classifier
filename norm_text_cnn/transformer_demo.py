#encoding=utf-8
import tensorflow as tf
import tcnn_data_helper,Multi_Head_self_attention_model
import transformer_encoder
#模型的超参数设置
tf.flags.DEFINE_integer("vector_size", 80, "每个单词向量的维度")
tf.flags.DEFINE_integer("psegs_size", 30, "每个单性向量的维度")
tf.flags.DEFINE_integer("sentence_length", 200, "设定的句子长度")
tf.flags.DEFINE_integer("num_filters", 128, "卷积核的个数")
tf.flags.DEFINE_integer("num_classes", 10, "类别种类数")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2正则化系数的比率")
filter_hs=[3,4,5]
tf.flags.DEFINE_float("keep_prob", 0.6, "丢失率")
tf.flags.DEFINE_integer("batch_size", 128, "每个批次的大小")
tf.flags.DEFINE_integer("num_epochs", 2, "训练的轮数")
tf.flags.DEFINE_integer("num_steps", 100, "学习率衰减的步数")
tf.flags.DEFINE_float("init_learning_rate", 0.05, "初始学习率")
#设定参数
tf.flags.DEFINE_string("model_save_location","textcnn_model/model", "网络模型的保存地址")
tf.flags.DEFINE_string("file_location","data/cut_pseg_train_data.csv", "数据文件的保存地址")
tf.flags.DEFINE_string("w2v_model_location","w2v_model/w2v_model", "词向量模型的保存地址")
tf.flags.DEFINE_string("words_location","words/words.txt", "训练数据词表的保存地址")
tf.flags.DEFINE_string("psegs_location","words/psegs.txt", "训练数据词性的保存地址")
FLAGS = tf.flags.FLAGS
x = tf.placeholder(tf.int32, [None, FLAGS.sentence_length], name='input')
y = tf.placeholder('float', [None, FLAGS.num_classes], name='output')
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
def backward_propagation():
    word_vecs, sen_index, psegs_vec, sen_pseg_index, one_hot_label=tcnn_data_helper.process_file(FLAGS.file_location,FLAGS.w2v_model_location,FLAGS.words_location,FLAGS.psegs_location,
                                                                        False,FLAGS.sentence_length,FLAGS.vector_size,FLAGS.psegs_size)
    #划分训练集和测试集
    print('begin 获得训练集测试集')
    X_train = sen_index[:-30]
    X_test = sen_index[-30:]
    y_train = one_hot_label[:-30]
    y_test = one_hot_label[-30:]
    #首先是embedding层获得词向量数据
    with tf.name_scope("embedding"):
        input_x = tf.nn.embedding_lookup(word_vecs,x)
        # 注意此处只能用tf.expand_dims()不能用np.expand_dims()，因为此处还没feed进去值
        # input_x = tf.expand_dims(input_x,-1)
    #初始化模型
    res = transformer_encoder.Encoder().build(input_x)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #批量获得数据
        for epoch in range(FLAGS.num_epochs):
            batch_train = tcnn_data_helper.batch_iter2(X_train, y_train, FLAGS.batch_size)
            total_batch = 0
            #一个批次大小的数据无法参与训练
            for x_batch,y_batch in batch_train:
                total_batch+=1
                feed_dict={x:x_batch,y:y_batch,keep_prob:FLAGS.keep_prob}
                aa=sess.run(res,feed_dict=feed_dict)
                print(aa.shape)
if __name__ == '__main__':
    backward_propagation()