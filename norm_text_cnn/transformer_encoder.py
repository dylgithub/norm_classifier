import tensorflow as tf
import Multi_Head_self_attention_model
import transformer_layer
class Encoder:
    """Encoder class"""
    #随后要进行批量归一化和添加残差网络，直接运行的元素相加，因此这里model_dim和词向量的维度要保持相同
    def __init__(self,
                 num_layers=1,
                 num_heads=6,
                 linear_key_dim=600,
                 linear_value_dim=600,
                 model_dim=80,
                 ffn_dim=100,
                 dropout=0.6):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
    #Transformer的实现
    def build(self, encoder_inputs):
        o1 = tf.identity(encoder_inputs)
        for i in range(1,self.num_layers+1):
            with tf.variable_scope("layer_%d" % i):
                #这里进行批量归一化和添加残差网络直接运行的元素相加，model_dim和词向量的维度要保持相同
                o2 = self._add_and_norm(o1,self._self_attention(q=o1,
                                                                 k=o1,
                                                                 v=o1), num=1)
                o3 = self._add_and_norm(o2,self._feed_forward(o2), num=2)
                o1 = tf.identity(o3)
        #输出的维度为[batch_size, seq_length, embedding_size]
        return o3
    #多头自注意力机制的实现，返回的是一个[batch_size,sentence_length,model_dim]
    def _self_attention(self, q, k, v):
        with tf.variable_scope("self-attention"):
            attention = Multi_Head_self_attention_model.Attention(num_heads=self.num_heads,
                                    masked=False,
                                    linear_key_dim=self.linear_key_dim,
                                    linear_value_dim=self.linear_value_dim,
                                    model_dim=self.model_dim,
                                    dropout=self.dropout)
            return attention.multi_head(q, k, v)
    #批量归一化和加入残差网络
    #第一层的x是[batch_size,sentence_length,vector_size]
    #sub_layer_x是多头自注意力机制的输出为[batch_size,sentence_length,model_dim]
    def _add_and_norm(self, x, sub_layer_x, num=0):
        with tf.variable_scope("add_and_norm_%d" % num):
            return tf.contrib.layers.layer_norm(tf.add(x,sub_layer_x))
    #实现feed_forward部分
    def _feed_forward(self, output):
        with tf.variable_scope("feed-forward"):
            ffn = transformer_layer.FFN(w1_dim=self.ffn_dim,
                      w2_dim=self.model_dim,
                      dropout=self.dropout)
            return ffn.dense_relu_dense(output)

