# coding=gbk

import pandas as pd
import numpy as np


def load_citeseer(path, train_nodes=140):
    raw_data = pd.read_csv(path+'/citeseer.content', sep='\t', header=None)
    raw_data_cites = pd.read_csv(path+'/citeseer.cites', sep='\t', header=None)

    # ��ȡ�ڽӾ���
    node_num = raw_data.shape[0]
    node_id = list(raw_data.index)
    paper_id = list(raw_data[0])
    paper_id = [str(a_paper_id) for a_paper_id in paper_id]
    map_dict = dict(zip(paper_id, node_id))

    A = np.eye(node_num, dtype='float32')
    for paper_id_i, paper_id_j in zip(raw_data_cites[0], raw_data_cites[1]):
        try:
            x = map_dict[paper_id_i]
            y = map_dict[paper_id_j]
            if x != y:
                A[x][y] = A[y][x] = 1
        except:
            print(f'{paper_id_i} or {paper_id_j} is not in map_dict.keys()!')

    # ��ȡ��������ͱ�ǩ����
    X = raw_data.iloc[:,1:-1].to_numpy(dtype='float32')
    Y = pd.get_dummies(raw_data[3704]).to_numpy(dtype='float32')

    idx_train = np.random.choice(A.shape[0], train_nodes)
    train_mask = np.array([1 if i in idx_train else 0 for i in range(A.shape[0])], dtype=np.bool8)
    val_mask = np.array([0 if i in idx_train else 1 for i in range(A.shape[0])], dtype=np.bool8)
    
    Y_train = np.zeros_like(Y)
    Y_val = np.zeros_like(Y)
    Y_train[train_mask, :] = Y[train_mask, :]
    Y_val[val_mask, :] = Y[val_mask, :]

    return A, X, Y_train, Y_val, train_mask, val_mask


def load_cora(path, train_nodes=140):
    """
    ��ȡcora���ݼ�
    ������
        - path�����ݼ�·��
        - train_nodes��ѵ���ڵ�����
    ���أ�
        - A���ڽӾ���
        - X����������
        - Y_train��ѵ��ʱ�ı�ǩ
        - Y_val����֤ʱ�ı�ǩ
        - train_mask��ѵ��ʱ��mask
        - val_mask����֤ʱ��maxk
    """

    # �ڵ�����
    raw_data = pd.read_csv(path+'/cora.content', sep='\t', header=None)
    # ��������
    raw_data_cites = pd.read_csv(path+'/cora.cites', sep='\t', header=None)
    node_num = raw_data.shape[0]
    node_id = list(raw_data.index)
    paper_id = list(raw_data[0])
    c = zip(paper_id, node_id)
    map_ = dict(c)
    A = np.zeros((node_num,node_num), dtype='float32')
    for paper_id_i, paper_id_j in zip(raw_data_cites[0], raw_data_cites[1]):
        x = map_[paper_id_i]
        y = map_[paper_id_j]
        A[x][y] = A[y][x] = 1
    X = raw_data.iloc[:,1:-1].to_numpy(dtype='float32')
    Y = pd.get_dummies(raw_data[1434]).to_numpy(dtype='float32')
    
    idx_train = np.random.choice(A.shape[0], train_nodes)
    train_mask = np.array([1 if i in idx_train else 0 for i in range(A.shape[0])], dtype=np.bool8)
    val_mask = np.array([0 if i in idx_train else 1 for i in range(A.shape[0])], dtype=np.bool8)
    
    Y_train = np.zeros_like(Y)
    Y_val = np.zeros_like(Y)
    Y_train[train_mask, :] = Y[train_mask, :]
    Y_val[val_mask, :] = Y[val_mask, :]

    return A, X, Y_train, Y_val, train_mask, val_mask