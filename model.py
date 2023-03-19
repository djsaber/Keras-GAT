# coding=gbk

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.regularizers import l2
from keras.layers import Layer, Dropout, LeakyReLU, Input
from keras.models import Model


class GraphAttentionLayer(Layer):
    """实现多头图注意力层
    参数：
        - hidden_dim：特征维度
        - attn_heads：注意力头数
        - attn_heads_reduction：多头处理方法，'concat'拼接处理，'average'平均处理
        - dropout_rate=0.5：dropout概率
        - mask：掩码
        - activation='relu'：激活函数
        - use_bias=True：是否使用偏置
        - kernel_initializer：权重初始化方法
        - kernel_regularizers：权重正则化方法
        - bias_regularizers：偏置正则化方法
        - attn_kernel_regularizers：注意力核权重正则化方法
        - activity_regularizers：激活函数正则化方法
        - kernel_constraint：权重约束方法
        - attn_kernel_constraint：注意力核权重约束方法
    """
    def __init__(
        self,
        hidden_dim,
        attn_heads,
        attn_heads_reduction='concat',
        dropout_rate=0.5,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zero',
        attn_kernel_initializer='glorot_uniform',
        kernel_regularizer=None,
        bias_regularizer=None,
        attn_kernel_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        attn_kernel_constraint=None,
        **kwargs
        ):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction method: concat, average')
        
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.dropout_rate = dropout_rate
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        
        if attn_heads_reduction == 'concat':
            self.output_dim = self.attn_heads * self.hidden_dim
        else:
            self.output_dim = self.hidden_dim

        self.kernels = []
        self.biases = []
        self.attn_kernels = []       

    def build(self, input_shape):
        super(GraphAttentionLayer, self).build(input_shape)
        input_dim = input_shape[0][-1]
        # 初始化每个head参数
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(
                name=f'kernel_{head}',
                shape=(input_dim, self.hidden_dim),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint
                )
            self.kernels.append(kernel)

            # Layer bias
            if self.use_bias:
                bias = self.add_weight(
                    name=f'bias_{head}',
                    shape=(self.hidden_dim, ),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint
                    )
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(
                name=f'attn_kernel_self_{head}',
                shape=(self.hidden_dim, 1),
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint
                )
            attn_kernel_neighs = self.add_weight(
                name=f'attn_kernel_neighs_{head}',
                shape=(self.hidden_dim, 1),
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint
                )
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])      

    def call(self, inputs):
        X = inputs[0]
        A = inputs[1]
        mask = -10e9 * (1.0 - A)
        outputs_heads = []
        for head in range(self.attn_heads):
            # (input_dim, feature_dim)
            kernel = self.kernels[head]
            # (2, feature_dim, 1)
            attn_kernel = self.attn_kernels[head]
            # (nodes, feature_dim)
            features = K.dot(X, kernel)
            # 计算feature combinations, (nodes, 1)
            attn_for_self = K.dot(features, attn_kernel[0])
            attn_for_neighs = K.dot(features, attn_kernel[1])
            # (nodes, nodes)
            dense = attn_for_self + K.transpose(attn_for_neighs)
            dense = LeakyReLU(alpha=0.2)(dense)
            # mask valus befor activate
            dense += mask
            # 计算attention系数(nodes, nodes)
            dense = K.softmax(dense)
            # dropout
            dropout_attn = Dropout(rate=self.dropout_rate)(dense)
            dropout_feat = Dropout(rate=self.dropout_rate)(features)
            # 更新节点特征(nodes, feature_dim)
            nodes_features = K.dot(dropout_attn, dropout_feat)

            if self.use_bias:
                nodes_features = K.bias_add(nodes_features, self.biases[head])          
            outputs_heads.append(nodes_features)

        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs_heads)
        else:
            output = K.mean(K.stack(outputs_heads), axis=0)
        
        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], self.output_dim)
        return output_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'attn_heads': self.attn_heads,
            'attn_heads_reduction': self.attn_heads_reduction,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'use_bias': self.use_bias,

            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'attn_kernel_initializer': self.attn_kernel_initializer,

            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'attn_kernel_regularizer': self.attn_kernel_regularizer,
            'activity_regularizers': self.activity_regularizers,

            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'attn_kernel_constraint': self.attn_kernel_constraint,

            'kernels': self.kernels,
            'biases': self.biases,
            'attn_kernels': self.attn_kernels
            })
        return config



class GAT(Model):

    def __init__(self, hidden_dim, output_dim, attn_heads, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = GraphAttentionLayer(
            hidden_dim=hidden_dim,
            attn_heads=attn_heads,
            attn_heads_reduction='concat',
            dropout_rate=dropout_rate,
            activation='elu',
            kernel_regularizer=l2(5e-4/2),
            attn_kernel_regularizer=l2(5e-4/2)
            )
        self.dropout = Dropout(dropout_rate)
        self.layer2 = GraphAttentionLayer(
            hidden_dim=output_dim,
            attn_heads=attn_heads,
            attn_heads_reduction='average',
            dropout_rate=dropout_rate,
            activation='softmax',
            kernel_regularizer=l2(5e-4/2),
            attn_kernel_regularizer=l2(5e-4/2)
            )

    def call(self, inputs):
        A = inputs[1]
        out = self.layer1(inputs)
        out = self.dropout(out)
        out = self.layer2([out, A])
        return out

    def build(self, input_shape):
        super().build(input_shape)
        self.call([Input(input_shape[0][1:]), Input(input_shape[1][1:])])