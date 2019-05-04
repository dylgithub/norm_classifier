import tensorflow as tf
class FFN:
    """FFN class (Position-wise Feed-Forward Networks)"""
    def __init__(self,
                 w1_dim=200,
                 w2_dim=100,
                 dropout=0.6):
        self.w1_dim = w1_dim
        self.w2_dim = w2_dim
        self.dropout = dropout
    #FFN(x)=max(0,xW1+b1)W2+b2
    def dense_relu_dense(self, inputs):
        output = tf.layers.dense(inputs, self.w1_dim, activation=tf.nn.relu)
        output =tf.layers.dense(output, self.w2_dim)
        return tf.nn.dropout(output,self.dropout)
    def conv_relu_conv(self):
        raise NotImplementedError("i will implement it!")
