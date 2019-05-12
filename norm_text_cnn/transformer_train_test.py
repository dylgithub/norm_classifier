#encoding=utf-8
import tensorflow as tf
import time
import os
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import tcnn_data_helper
from transformer_model import TransformerModel
from transformer_config import Config
#网络模型的保存文件夹
model_save_location="checkpoints/transformer"
#网络模型保存的相对路径
save_path = os.path.join(model_save_location, 'best_model')
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
def train_tes_model(model,transformer_config):
    # 配置Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/transformer'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    #删除原来已存在的tensorboard文件
    else:
        file_list=os.listdir(tensorboard_dir)
        if len(file_list)>0:
            for file in file_list:
                os.remove(os.path.join(tensorboard_dir,file))
    tf.summary.scalar("loss", model.losses)
    tf.summary.scalar("accuracy", model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    # 配置 Saver，用以保存模型
    saver = tf.train.Saver()
    if not os.path.exists(model_save_location):
        os.makedirs(model_save_location)
    #获得训练数据和测试数据
    start_time = time.time()
    _,sen_index,_,sen_pseg_index,one_hot_label = tcnn_data_helper.process_file(
        transformer_config.file_location,transformer_config.w2v_model_location,transformer_config.words_location,transformer_config.psegs_location,
        False,transformer_config.sentence_length,transformer_config.vector_size,transformer_config.pseg_size)
    X_train,X_test,X_pseg_train,X_pseg_test,y_train,y_test=train_test_split(sen_index,sen_pseg_index,one_hot_label,test_size=0.1)
    time_dif = get_time_dif(start_time)
    print("load data usage:",time_dif)
    print('Training and Testing...')
    start_time = time.time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        writer.add_graph(sess.graph)
        #批量获得数据
        for epoch in range(transformer_config.num_epochs):
            batch_train = tcnn_data_helper.batch_iter(X_train,X_pseg_train,y_train,transformer_config.batch_size)
            total_batch = 0
            for x_batch,x_pseg_batch,y_batch in batch_train:
                total_batch+=1
                feed_dict={model.input_x:x_batch,model.input_x_pseg:x_pseg_batch,model.input_y:y_batch,model.keep_prob:transformer_config.keep_prob}
                if total_batch%transformer_config.save_per_batch==0:
                    summary_str = sess.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(summary_str,total_batch)  # 将summary 写入文件
                if total_batch%transformer_config.print_per_batch == 0:
                    train_accuracy = model.accuracy.eval(feed_dict=feed_dict)
                    print("Epoch %d:Step %d accuracy is %f" % (epoch+1,total_batch,train_accuracy))
                sess.run(model.optim, feed_dict=feed_dict)
        saver.save(sess,save_path)
        #训练完之后通过测试集测试模型
        batch_train = tcnn_data_helper.batch_iter(X_test, X_pseg_test,y_test,transformer_config.batch_size)
        all_test_pred=[]
        for x_batch, x_pseg_batch, y_batch in batch_train:
            test_pred=model.pred_label.eval(feed_dict={model.input_x:x_batch,model.input_x_pseg:x_pseg_batch,model.input_y:y_batch,model.keep_prob:1.0})
            all_test_pred.extend(test_pred)
        test_label = np.argmax(y_test,1)
        #要和id所代表的类别标签顺序相同
        categories=['教育','时尚', '家居', '娱乐', '财经', '体育','房产','游戏','时政','科技']
        # 评估
        print("Precision, Recall and F1-Score...")
        print(classification_report(test_label,all_test_pred,target_names=categories))
        # 混淆矩阵
        print("Confusion Matrix...")
        cm = confusion_matrix(test_label,all_test_pred)
        print(cm)
        time_dif = get_time_dif(start_time)
        print("train_and_test usage:", time_dif)
if __name__ == '__main__':
    transformer_config=Config()
    model=TransformerModel(transformer_config)
    train_tes_model(model,transformer_config)