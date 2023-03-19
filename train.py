# coding=gbk

from model import GAT
from utils import *
from keras.optimizers import Adam
import os


#---------------------------------设置参数-------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
data_set = 'cora'                           # 数据集 'cora' 或者 'citeseer'
train_nodes = 140                           # 训练节点数量
epochs = 100                                # 迭代次数
hidden_dim=128                              # 隐层维度
if data_set=='cora': output_dim=7           # 输出维度
if data_set=='citeseer': output_dim=6                         
att_heads=6                                 # 注意力头数量
dropout_rate = 0.5                          # dropout概率率
LR = 5e-3                                   # 学习率
#-----------------------------------------------------------------------------


#---------------------------------设置路径-------------------------------------
data_path = "D:/科研/python代码/炼丹手册/GAT/datasets/"
save_path = "D:/科研/python代码/炼丹手册/GAT/save_models/gat.h5"
#-----------------------------------------------------------------------------


#--------------------------------加载数据集-------------------------------------
if data_set == 'cora':
    A, X, Y_train, Y_val, train_mask, val_mask = load_cora(data_path+data_set, train_nodes)
if data_set == 'citeseer':
    A, X, Y_train, Y_val, train_mask, val_mask = load_citeseer(data_path+data_set, train_nodes)
validation_data = ([X, A], Y_val, val_mask)
print(f'加载数据集：\n邻接矩阵：{A.shape}\n特征矩阵：{X.shape}')
#-----------------------------------------------------------------------------


#---------------------------------搭建模型-------------------------------------
gat = GAT(
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    attn_heads=att_heads,
    dropout_rate=dropout_rate
    )
gat.build(input_shape=[(None, X.shape[-1]), (None, None)])
gat.summary()
gat.compile(
    optimizer=Adam(LR), 
    loss='categorical_crossentropy',
    weighted_metrics=['acc']
    )
#-----------------------------------------------------------------------------


#--------------------------------训练和保存-------------------------------------
history = gat.fit(
    x=[X, A],
    y=Y_train,
    batch_size=A.shape[0],
    epochs=100,
    sample_weight=train_mask,
    validation_data=validation_data,
    shuffle=False
    )
gat.save_weights(save_path)
#-----------------------------------------------------------------------------