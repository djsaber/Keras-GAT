# coding=gbk

from model import GAT
from utils import *
from keras.optimizers import Adam
import os


#---------------------------------���ò���-------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
data_set = 'cora'                           # ���ݼ� 'cora' ���� 'citeseer'
train_nodes = 140                           # ѵ���ڵ�����
epochs = 100                                # ��������
hidden_dim=128                              # ����ά��
if data_set=='cora': output_dim=7           # ���ά��
if data_set=='citeseer': output_dim=6                         
att_heads=6                                 # ע����ͷ����
dropout_rate = 0.5                          # dropout������
LR = 5e-3                                   # ѧϰ��
#-----------------------------------------------------------------------------


#---------------------------------����·��-------------------------------------
data_path = "D:/����/python����/�����ֲ�/GAT/datasets/"
save_path = "D:/����/python����/�����ֲ�/GAT/save_models/gat.h5"
#-----------------------------------------------------------------------------


#--------------------------------�������ݼ�-------------------------------------
if data_set == 'cora':
    A, X, Y_train, Y_val, train_mask, val_mask = load_cora(data_path+data_set, train_nodes)
if data_set == 'citeseer':
    A, X, Y_train, Y_val, train_mask, val_mask = load_citeseer(data_path+data_set, train_nodes)
validation_data = ([X, A], Y_val, val_mask)
print(f'�������ݼ���\n�ڽӾ���{A.shape}\n��������{X.shape}')
#-----------------------------------------------------------------------------


#---------------------------------�ģ��-------------------------------------
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


#--------------------------------ѵ���ͱ���-------------------------------------
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