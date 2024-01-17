import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from models.CNN import G_model_1, G_model_2
from models.graph_convolution import GraphConvolution
from models.RRF import RandomReceptiveField
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout, Flatten, Reshape, Activation, Lambda
from tensorflow.keras.models import Model
from data_loader import Paderborn_dataset
import numpy as np
from tensorflow.keras.utils import to_categorical
import scipy.io as sio
from losses.scl_loss import SCL_loss
from losses.cov_loss import covariance_loss
from batch_data import get_balanced_batch
from tensorflow.keras.layers import MultiHeadAttention
import os
from sklearn.preprocessing import StandardScaler
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
random_seed = 1388
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)


# The solver for training and testing
class CustomModel():
    def __init__(self):
        self.ndomain = 3
        self.nclasses = 3
        self.nfeat = 1024
        self.input_shape = (1, 4096, 1)
        self.sigma = 0.005
        self.len_segment = 4096
        self.len_data = 1
        self.dropout_rate = 0.5
        self.beta = 0.7
        self.entropy_threshold = 0.1
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.batch_size = 24
        self.augment = 2

        # define the feature extractor and GCN-based classifier
        self.conv_1 = GraphConvolution(self.nfeat, 256)
        self.conv_2 = GraphConvolution(256, self.nclasses)
        self.mean = tf.Variable(tf.zeros((self.augment*self.nclasses * self.ndomain, self.nfeat)), trainable=False)
        self.adj = tf.Variable(tf.zeros((self.augment*self.nclasses * self.ndomain, self.augment*self.nclasses * self.ndomain)), trainable=False)

        # 使用 Model 对象
        self.model_1 = G_model_1((self.len_data, self.len_segment, 1))
        self.model_2 = G_model_2((self.len_data, self.len_segment, 1))

    # compute the Euclidean distance between two tensors
    def euclid_dist(self, x, y):
        x_sq = tf.reduce_mean(tf.square(x), axis=-1)
        x_sq_ = tf.tile(tf.expand_dims(x_sq, axis=1), [1, tf.shape(y)[0]])
        y_sq = tf.reduce_mean(tf.square(y), axis=-1)
        y_sq_ = tf.tile(tf.expand_dims(y_sq, axis=0), [tf.shape(x)[0], 1])
        xy = tf.matmul(x, y, transpose_b=True) / tf.cast(tf.shape(x)[-1], tf.float32)
        dist = x_sq_ + y_sq_ - 2 * xy
        return dist

    # construct the extended adjacency matrix
    def construct_adj(self, inp):
        mean_new = self.mean
        dist = self.euclid_dist(mean_new, inp)  # 调用 euclid_dist 函数计算 prototypes (self.mean) 和 embeddings 的距离, 9×192
        sim = tf.exp(-dist / (2 * self.sigma ** 2))   # 使用指数函数, 计算基于距离的相似度, radial basis function (RBF) kernel, 9×192
        # tf.print(tf.shape(sim))
        E = tf.eye(tf.shape(inp)[0])  # 创建一个单位矩阵 E，其大小与 feats 的样本数量相等
        adj_new = self.adj
        A = tf.concat([adj_new, sim], axis=1)  # 9×201
        B = tf.concat([tf.transpose(sim), E], axis=1)  # 192×201
        gcn_adj = tf.concat([A, B], axis=0)  # 使用 tf.concat 来组合 self.adj（原始邻接矩阵）、sim（相似度矩阵）和 E，以构建扩展的邻接矩阵 gcn_adj
        out = gcn_adj  # 201×201
        return out

    def update_statistics(self, feats, labels):
        batch_size = self.batch_size
        # tf.print(feats)
        label = tf.concat([labels, labels], axis=0)
        domain_size = self.augment*self.ndomain
        i = 0
        curr_mean = tf.TensorArray(tf.float32, size=self.nclasses * domain_size, dynamic_size=False,
                                   clear_after_read=False)
        # print(domain_size)
        for domain_idx in range(domain_size):
            # print(domain_idx)
            for class_idx in range(self.nclasses):
                # print(class_idx)
                start_idx = domain_idx * batch_size
                end_idx = start_idx + batch_size
                domain_feats = feats[start_idx:end_idx, :]
                domain_labels = label[start_idx:end_idx,]
                # tf.print('tf.shape(domain_labels): ', domain_labels)

                class_feats = domain_feats[tf.squeeze(domain_labels) == class_idx]
                mean_feat = tf.reduce_mean(class_feats, axis=0)
                # tf.print('tf.shape(mean_feat): ', class_feats)
                curr_mean = curr_mean.write(i, mean_feat)
                i = i+1

        curr_mean = curr_mean.stack()
        curr_mean_new = curr_mean
        curr_mask = tf.cast(tf.reduce_sum(curr_mean_new, axis=-1) != 0, dtype=tf.float32)[:, tf.newaxis]
        # self_mean = self.mean
        new_mean = self.mean * (1 - curr_mask) + (self.mean * self.beta + curr_mean_new * (1 - self.beta)) * curr_mask
        self.mean.assign(new_mean)

        curr_dist = self.euclid_dist(new_mean, new_mean)
        new_adj = tf.exp(-curr_dist / (2 * self.sigma ** 2))
        self.adj.assign(new_adj)
        for i in range(self.augment*self.ndomain):
            for j in range(self.augment*self.ndomain):
                start_i = i * self.nclasses
                end_i = (i + 1) * self.nclasses
                start_j = j * self.nclasses
                end_j = (j + 1) * self.nclasses
                adj_ij = self.adj[start_i:end_i, start_j:end_j]
                # 创建一个相同形状的掩码，对角线为0，其他位置为1
                mask = tf.eye(self.nclasses)
                # 应用掩码，将非对角线上的元素限制在 1e-3 以下
                tensor_masked = tf.where((adj_ij > 1e-3) & (mask == 0), 1e-3, adj_ij)
                self.adj[start_i:end_i, start_j:end_j].assign(tensor_masked)
        loss_local = tf.reduce_sum(tf.reduce_mean(tf.square((curr_mean_new - new_mean) * curr_mask), axis=-1)) / tf.cast(tf.shape(feats)[0], tf.float32)
        return loss_local

    def build_scl(self):
        inp = Input(shape=(self.nfeat,))
        x = inp
        x = tf.concat([self.mean, x], axis=0)
        x = Dropout(0.3)(x)
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        out = x
        return Model(inp, out)

    def G_model(self):
        inp = Input(shape=(self.len_data, self.len_segment, 1))
        out1 = self.model_1(inp)
        out1 = RandomReceptiveField(8, 0, 7)(out1)
        out1 = Flatten()(out1)

        out2 = self.model_2(inp)
        out2 = RandomReceptiveField(8, 0, 7)(out2)
        out2 = Flatten()(out2)

        out = tf.concat([out1, out2], axis=0)
        return Model(inp, out)

    def GCN_model(self):
        inp = Input(shape=(self.nfeat,))
        x = inp
        self_mean = self.mean
        x1 = tf.concat([self_mean, x], axis=0)  # x1: 201×1024, self.mean: prototype, inp: embeddings
        x2 = Lambda(self.construct_adj)(x)  # x2: 201×201
        x0 = x1
        adj = x2
        x0 = self.conv_1([x0, adj])  # W.x1.adj
        out0 = x0
        x0 = Activation('relu')(x0)
        x0 = Dropout(self.dropout_rate)(x0)
        x0 = self.conv_2([x0, adj])
        out = tf.concat([x1, x0, out0], axis=1)
        return Model(inp, out)

    def global_loss(self, y_label, out):
        feats = out[tf.shape(self.mean)[0]:, :self.nfeat]
        gcn_logit = out[:, self.nfeat:self.nfeat+self.nclasses]
        gcn_logit = tf.nn.softmax(gcn_logit, axis=1)
        cce = tf.keras.losses.SparseCategoricalCrossentropy()
        domain_logit = gcn_logit[:tf.shape(self.mean)[0], :]
        domain_label = tf.tile(tf.range(self.nclasses), [self.augment*self.ndomain])  # domain_label 是一个包含类别标签的张量，
        loss_cls_dom = cce(domain_label, domain_logit)

        cov = covariance_loss(feats[:self.ndomain*self.batch_size, :], feats[self.ndomain*self.batch_size:, :])

        total_loss = loss_cls_dom + 0.0001*cov
        return total_loss

    def scl_loss(self, y_label, out):
        feats_1 = out[tf.shape(self.mean)[0]:tf.shape(self.mean)[0] + self.batch_size * self.ndomain, :]
        feats_2 = out[tf.shape(self.mean)[0] + self.batch_size * self.ndomain:, :]
        y_label = tf.cast(y_label, tf.int32)
        loss_scl_1 = SCL_loss(y_label, feats_1)
        loss_scl_2 = SCL_loss(y_label, feats_2)
        return loss_scl_1 + loss_scl_2

    def train_step(self, train_data_1, train_label_1, train_data_2, train_label_2, train_data_3, train_label_3, test_data, test_label):
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        ms = self.G_model()
        ms.compile(optimizer=optimizer)
        GCN = self.GCN_model()
        GCN.compile(optimizer=optimizer, loss=self.global_loss)
        scl = self.build_scl()
        scl.compile(optimizer=optimizer, loss=self.scl_loss)

        inp = Input(shape=(self.len_data, self.len_segment, 1))
        fea = ms(inp)
        out1 = GCN(fea)
        out2 = scl(fea)  # 判别训练数据和测试数据的特征相似度
        mt_p_c_d = Model(inp, [out1, out2])
        mt_p_c_d.compile(optimizer=optimizer, loss=[self.global_loss, self.scl_loss], loss_weights=[1, 1])

        epoch_size1 = len(train_data_1)
        batch_size = self.batch_size
        epochs = 20

        for i in range(epochs):
            print('i =', i)
            mt_p_c_d.trainable = True
            for j in range(int(epoch_size1 / batch_size)):
                s1_data, s1_label = get_balanced_batch(train_data_1, train_label_1, self.nclasses, self.batch_size)
                s2_data, s2_label = get_balanced_batch(train_data_2, train_label_2, self.nclasses, self.batch_size)
                s3_data, s3_label = get_balanced_batch(train_data_3, train_label_3, self.nclasses, self.batch_size)
                g_data = np.concatenate([s1_data, s2_data, s3_data], axis=0)
                g_label = np.concatenate([s1_label, s2_label, s3_label], axis=0)
                g_fea = ms(g_data)
                aaaa = self.update_statistics(g_fea, g_label)
                mt_p_c_d.train_on_batch(g_data, [g_label, g_label])
            num = 0
            sum = 0
            mt_p_c_d.trainable = False
            print(self.adj)
            print(self.mean)
            for j in range(int(len(test_data)/batch_size)):
                start_idx = j * batch_size
                tt_data = test_data[start_idx:(start_idx + batch_size), :]
                tt_label = test_label[start_idx:(start_idx + batch_size), ]
                test_fea = ms(tt_data)
                test_r = GCN(test_fea)
                ypred = test_r[self.mean.shape[0]:self.mean.shape[0] + batch_size, self.nfeat:self.nfeat+self.nclasses]
                ypred = np.argmax(ypred, axis=1)
                dd = np.argwhere(ypred == tt_label)
                num = num + len(dd)
                sum = sum + batch_size
            acc = num / sum
            print('testing accuracy: ', acc)


if __name__ == '__main__':
    model = CustomModel()

    scaler_domain1 = StandardScaler()
    scaler_domain2 = StandardScaler()
    scaler_domain3 = StandardScaler()
    scaler_domain4 = StandardScaler()

    data1, label1, data2, label2, data3, label3, data4, label4 = Paderborn_dataset()
    train_data_1 = data1
    train_data_1 = scaler_domain1.fit_transform(train_data_1)
    train_data_1 = train_data_1.reshape(len(train_data_1), 1, 4096, 1)
    train_label_1 = label1
    train_data_2 = data2
    train_data_2 = scaler_domain2.fit_transform(train_data_2)
    train_data_2 = train_data_2.reshape(len(train_data_2), 1, 4096, 1)
    train_label_2 = label2
    train_data_3 = data3
    train_data_3 = scaler_domain3.fit_transform(train_data_3)
    train_data_3 = train_data_3.reshape(len(train_data_3), 1, 4096, 1)
    train_label_3 = label3
    tt_data = data4
    tt_data = scaler_domain4.fit_transform(tt_data)
    tt_data = tt_data.reshape(len(tt_data), 1, 4096, 1)
    tt_label = label4

    model.train_step(train_data_1, train_label_1, train_data_2, train_label_2,
                     train_data_3, train_label_3, tt_data, tt_label)







