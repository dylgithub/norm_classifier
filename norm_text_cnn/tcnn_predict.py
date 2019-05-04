#encoding=utf-8
from tcnn_config import Config
from tcnn_data_helper import get_word_and_pseg2id_dict
import tensorflow.contrib.keras as kr
import tensorflow as tf
import jieba.posseg as pseg
import os
#设置训练好的模型的存储路径
save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_model')  # 最佳验证结果保存路径
class TCNNModel:
    def __init__(self):
        #重建图
        self.graph=tf.Graph()
        #把这个图设置为TensorFlow运行环境的默认图
        with self.graph.as_default():
            #导入已训练好的图模型
            self.saver=tf.train.import_meta_graph(save_path+'.meta')
        #获得预测中需要操作的参数
        self.keep_prob=self.graph.get_tensor_by_name("keep_prob:0")
        self.input_x=self.graph.get_tensor_by_name("input_x:0")
        self.input_x_pseg=self.graph.get_tensor_by_name("input_x_pseg:0")
        self.pred_label=self.graph.get_tensor_by_name("full_connection/ArgMax:0")
        #创建会话并指定会话启动的图
        self.session=tf.Session(graph=self.graph)
        #指定默认的session和图结构，并通过with关键字使得加载的模型添加到相应的默认图结构中
        with self.session.as_default():
            with self.graph.as_default():
                self.saver.restore(sess=self.session,save_path=save_path)
        self.config=Config()
        _,self.word2id_dict,_,self.pseg2id_dict=get_word_and_pseg2id_dict(self.config.words_location,self.config.psegs_location)
    def process_data(self,message):
        content_id_list=[]
        pseg_id_list=[]
        for w in pseg.cut(message):
            if w.word in self.word2id_dict:
                content_id_list.append(self.word2id_dict[w.word])
            else:
                content_id_list.append(self.word2id_dict['UNKNOW'])
            if w.flag[0] in ['v','n','l','y','r']:
                _key=w.flag[0]
            else:
                _key='x'
            pseg_id_list.append(self.pseg2id_dict[_key])
        #传入的第一个参数是一个List其中的元素是一个个小List，小List是把文本中的词转换为id
        #第二个参数是句子的最大长度，默认过长从前面截取，过短从前面填充0
        input_x=kr.preprocessing.sequence.pad_sequences([content_id_list], self.config.sentence_length)
        input_x_pseg=kr.preprocessing.sequence.pad_sequences([pseg_id_list], self.config.sentence_length)
        return input_x,input_x_pseg
    def do_predict(self,message):
        self._input_x,self._input_x_pseg=self.process_data(message)
        feed_dict = {
            self.input_x:self._input_x,
            self.input_x_pseg:self._input_x_pseg,
            self.keep_prob:1.0
        }
        y_pred_cls = self.session.run(self.pred_label, feed_dict=feed_dict)
        print(y_pred_cls)
if __name__ == '__main__':
    tcnn_model = TCNNModel()
    for i in range(2):
        str=input("Enter your input:")
        tcnn_model.do_predict(str)
    # test_demo = ['三星ST550以全新的拍摄方式超越了以往任何一款数码相机',
    #              '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00']
    # for i in test_demo:
    #     tcnn_model.do_predict(i)