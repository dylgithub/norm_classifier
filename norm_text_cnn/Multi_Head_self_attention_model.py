
import numpy as np
import tensorflow as tf



__all__ = [
    "positional_encoding", "Attention"
]

#在词向量上添加位置信息
def positional_encoding(dim, sentence_length, dtype=tf.float32):
    '''
    :param dim: 论文中提到除的是模型的维度，而根据encoder部分可知模型的维度必须和词向量的维度相同（因为要添加残差网络）故这里和传入词向量的维度效果相同
    最后得到的代表位置的向量均可和词向量直接相加
    :param sentence_length:
    :param dtype:
    :return:
    '''
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)
class Attention:
    """Attention class"""
    def __init__(self,
                 num_heads=6,
                 masked=False,
                 linear_key_dim=600,
                 linear_value_dim=600,
                 model_dim=200,
                 dropout=0.6):
        '''
        :param num_heads: 头的个数
        :param masked: 是否进行masked，分类中只用到encoder不需要masked
        :param linear_key_dim: 所有头concat的维度，必须是num_heads的整数倍
        :param linear_value_dim: 和上个参数相同
        :param model_dim: 经过多头自注意力机制之后进而进行线性变换，最后输出最后一维的维度
        :param dropout:
        '''
        #随后要对每个头进行维度的划分，因此这里必须能整除，否则抛异常
        assert linear_key_dim % num_heads == 0
        assert linear_value_dim % num_heads == 0
        self.num_heads = num_heads
        self.masked = masked
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.dropout = dropout
    def multi_head(self, q, k, v):
        q, k, v = self._linear_projection(q, k, v)
        #划分后每个维度为[batch_size,num_heads,max_seq_len,dim]
        qs, ks, vs = self._split_heads(q, k, v)
        outputs = self._scaled_dot_product(qs, ks, vs)
        #这里的维度为[batch_size,sentence_length,num_heads*key_dim_per_head==linear_key_dim]
        output = self._concat_heads(outputs)
        output = tf.layers.dense(output, self.model_dim)
        #最后返回的数据的维度为[batch_size,sentence_length,model_dim]
        #因为随后要进行批量归一化和添加残差网络直接运行的元素相加，所以model_dim要和词向量的维度保持相同
        return tf.nn.dropout(output,self.dropout)
    #线性变换部分，这里使用线性变换因为随后要使用多头进行头的划分
    def _linear_projection(self, q, k, v):
        q = tf.layers.dense(q,self.linear_key_dim, use_bias=False)
        k = tf.layers.dense(k,self.linear_key_dim, use_bias=False)
        v = tf.layers.dense(v,self.linear_value_dim, use_bias=False)
        return q, k, v
    #划分成不同的头，使得不同的头提取不同类型的特征
    def _split_heads(self, q, k, v):
        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            t_shape = tensor.get_shape().as_list()
            tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, dim // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, num_heads, max_seq_len, dim]
        qs = split_last_dimension_then_transpose(q, self.num_heads, self.linear_key_dim)
        ks = split_last_dimension_then_transpose(k, self.num_heads, self.linear_key_dim)
        vs = split_last_dimension_then_transpose(v, self.num_heads, self.linear_value_dim)
        return qs, ks, vs
    #实现缩放点积
    def _scaled_dot_product(self, qs, ks, vs):
        #每个头的维度
        key_dim_per_head = self.linear_key_dim // self.num_heads
        #qs,ks都为[batch_size,nb_head,sentence_length,dim]
        # 这里得到的o1是（batch_size,nb_head,sentence_length，sentence_length）
        #相当于各个头batch_size条数据进行(sentence_length，dim)*(dim，sentence_length）相当于相似度的计算
        o1 = tf.matmul(qs,ks,transpose_b=True)
        o2 = o1 / (key_dim_per_head**0.5)
        #实现文本分类只需用encoder故用不到masked
        if self.masked:
            diag_vals = tf.ones_like(o2[0, 0, :, :]) # (batch_size, num_heads, query_dim, key_dim)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (q_dim, k_dim)
            masks = tf.tile(tf.reshape(tril, [1, 1] + tril.get_shape().as_list()),
                            [tf.shape(o2)[0], tf.shape(o2)[1], 1, 1])
            paddings = tf.ones_like(masks) * -1e9
            o2 = tf.where(tf.equal(masks, 0), paddings, o2)
        #通过归一化获得每个时刻的注意力权重
        o3 = tf.nn.softmax(o2)
        # o3是（batch_size,nb_head，sentence_length，sentence_length）
        # vs是（batch_size, nb_head, sentence_length，size_per_head）
        # 返回值是（batch_size, nb_head, sentence_length，size_per_head）
        #相当于各个头batch_size条数据进行(sentence_length，sentence_length)*(sentence_length，size_per_head）
        #得到注意力值
        return tf.matmul(o3,vs)
    #concat各个头
    def _concat_heads(self, outputs):
        def transpose_then_concat_last_two_dimenstion(tensor):
            tensor = tf.transpose(tensor, [0,2,1,3]) # [batch_size,sentence_length,nb_head,dim]
            t_shape = tensor.get_shape().as_list()
            num_heads, dim = t_shape[-2:]
            return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])
        # 返回的维度是[batch_size,sentence_length,nb_head * size_per_head]
        return transpose_then_concat_last_two_dimenstion(outputs)
if __name__ == '__main__':
    data=positional_encoding(80,100)
    print(data.shape)
