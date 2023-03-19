# Keras-GAT
基于Keras搭建一个GAT，用cora数据集和citeseer数据集对GAT进行训练，完成节点分类测试。


环境：<br />
CUDA：11.6.134<br />
cuDNN：8.4.0<br />
keras：2.9.0<br />
tensorflow：2.9.1<br /><br />

注意：<br />
项目内目录中两个文件夹：<br />
1. /datasets：将数据集文件解压至此<br />
2. /save_models：保存训练好的模型权重文件，包括生成器权重和判别器权重两个文件<br />

GAT概述<br />
图神经网络(Graph Neural Network, GNN)是指神经网络在图上应用的模型的统称，根据采用的技术不同和分类方法的不同，
又可以分为下图中的不同种类，例如从传播的方式来看，图神经网络可以分为图卷积神经网络（GCN），图注意力网络（GAT），Graph LSTM等等<br />
图注意力网络(Graph Attention Network, GAT)，一种基于图结构数据的新型神经网络架构，利用隐藏的自我注意层来解决之前基于图卷积或其近似的方法的不足。通过堆叠层，节点能够参与到邻居的特征，可以(隐式地)为邻域中的不同节点指定不同的权值，而不需要任何代价高昂的矩阵操作(如反转)，也不需要预先知道图的结构。通过这种方法，该模型克服了基于频谱的故神经网络的几个关键挑战，并使得模型适用于归纳和推理问题。<br /><br />

数据集：<br />
cora：包含2708篇科学出版物网络，共有5429条边，总共7种类别。<br />
数据集中的每个出版物都由一个 0/1 值的词向量描述，表示字典中相应词的缺失/存在。 该词典由 1433 个独特的词组成。<br />
链接：https://pan.baidu.com/s/1u7v3oJcTvnFWAhHdSLHwtA?pwd=52dl 提取码：52dl<br />

citeseer：包含3312个节点，4723条边构成的引文网络。标签共6个类别。数据集的特征维度是3703维。<br />
链接：https://pan.baidu.com/s/11n2AQCVSV6OevSkUhYWcNg?pwd=52dl 提取码：52dl<br /><br />

通过测试，采用以下设置：<br />
train_nodes = 140                           # 训练节点数量<br />
epochs = 100                                # 迭代次数<br />
hidden_dim=128                              # 隐层维度<br />            
att_heads=6                                 # 注意力头数量<br />
dropout_rate = 0.5                          # dropout概率率<br />
Adam LR = 5e-3                              # 学习率<br />
GAT在cora数据集和citeseer数据集上具有70%和80%左右的准确率，上面参数随便设置的，调好超参数应该还能提高一点。
