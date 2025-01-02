"""
Script for ENGINE: Enhancing Neuroimaging and Genetic Information by Neural Embedding framework
Written in Tensorflow 2.1.0
"""

# Import APIs
import tensorflow as tf
import numpy as np
from keras.layers import MultiHeadAttention, concatenate
from sklearn.metrics.pairwise import cosine_similarity

class engine(tf.keras.Model):   
    tf.keras.backend.set_floatx('float32') 
    """ENGINE framework"""

    def __init__(self, N_o):
        super(engine, self).__init__() 
        self.N_o = N_o # the number of classification outputs
        print (self)
        """SNP Representation Module"""
        # Encoder network, Q
        self.encoder = tf.keras.Sequential(  
            [
                tf.keras.layers.InputLayer(input_shape=(2703,)),  # F_SNP = 2703
                tf.keras.layers.Dense(units=500, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dense(units=200, activation=None, kernel_regularizer='L1L2'),  # 2 * dim(z_SNP) = 100
            ]
        )

        # Decoder network, P
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(100,)), # dim(z_SNP) = 100
                tf.keras.layers.Dense(units=500, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dense(units=2703, activation=None, kernel_regularizer='L1L2'), # F_SNP = 4946 2703
            ]
        )
        """MRI Representation Module"""
        # Encoder network, MRI share
        self.encoder_MRI = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(116,)),  # F_MRI = 116
                tf.keras.layers.Dense(units=200, activation='elu', kernel_regularizer='L1L2'),  # 200神经元 elu激活函数 L1L2正则化
                tf.keras.layers.Dense(units=100, activation='elu', kernel_regularizer='L1L2'),  # 2 * dim(z_MRI) = 100
            ]
        )

        self.decoder_MRI = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(100,)),  # dim(z_SNP) = 100
                tf.keras.layers.Dense(units=200, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dense(units=116, activation=None, kernel_regularizer='L1L2'),  # F_SNP = 2703
            ]
        )
        """PET Representation Module"""
        # feature_columns = [SparseFeat('sparse_feature_1', vocabulary_size=10, embedding_dim=4),DenseFeat('dense_feature_1', 1),
        #     # 添加其他的特征列...
        # ]
        # Encoder network, PET share
        self.encoder_PET = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(116,)),  # F_PET = 116
                tf.keras.layers.Dense(units=200, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dense(units=100, activation='elu', kernel_regularizer='L1L2'),  # 2 * dim(z_PET) = 100
            ]
        )

        self.decoder_PET = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(100,)),  # dim(z_SNP) = 100
                tf.keras.layers.Dense(units=200, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dense(units=116, activation=None, kernel_regularizer='L1L2'),  # F_SNP = 2703
            ]
        )
        """Attentive Vector Generation Module"""

        self.asso_MRI = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(100+4,)),  # dim(z) = dim(z_SNP) + dim(c)
                tf.keras.layers.Dense(units=200, activation='elu', kernel_regularizer='L1L2'), 
                tf.keras.layers.Dense(units=332, activation=None, kernel_regularizer='L1L2'),  
            ]
        )

        
        ## 参数共享生成网络
        # self.asso_PET_MRI = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.InputLayer(input_shape=(104,)),  
        #         tf.keras.layers.Dense(units=100, activation='elu', kernel_regularizer='L1L2'),
        #     ]
        # )
        # self.asso_PET = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.InputLayer(input_shape=(100,)),  
        #         tf.keras.layers.Dense(units=116, activation='sigmoid', kernel_regularizer='L1L2'),
        #     ]
        # )
        # self.asso_MRI = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.InputLayer(input_shape=(100,)),  
        #         tf.keras.layers.Dense(units=116, activation='sigmoid', kernel_regularizer='L1L2'),
        #     ]
        # )
        # self.asso_MASK = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.InputLayer(input_shape=(100,)),  
        #         tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_regularizer='L1L2'),
        #     ]
        # )

        # Discriminator network, D
        self.discriminator_MRI = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(116,)), 
                tf.keras.layers.Dense(units=50, activation='relu', kernel_regularizer='L1L2'), # real or fake
                # tf.keras.layers.Dense(units=50, kernel_regularizer='L1L2'),
                # tf.keras.layers.LeakyReLU(alpha=0.01),
                tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer='L1L2'), # real or fake
            ]
        )

        self.discriminator_PET = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(116,)), 
                tf.keras.layers.Dense(units=50, activation='relu', kernel_regularizer='L1L2'), # real or fake
                # tf.keras.layers.Dense(units=50, kernel_regularizer='L1L2'),
                # tf.keras.layers.LeakyReLU(alpha=0.01),
                tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer='L1L2'), # real or fake
            ]
        )



        """Diagnostician Module"""
        # Diagnostician network, C
        self.diagnostician_share = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(100,)), 
                tf.keras.layers.Dense(units=50, activation='elu', kernel_regularizer='L1L2'),
            ]
        )

        self.diagnostician_clf = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(50,)),  
                # tf.keras.layers.Dropout(rate=0.2),  
                # tf.keras.layers.Dense(units=200, activation=tf.nn.leaky_relu),  
                # tf.keras.layers.Dropout(rate=0.2),  
                # tf.keras.layers.Dense(units=100, activation=tf.nn.leaky_relu),  
                tf.keras.layers.Dense(units=self.N_o, activation='Softmax', kernel_regularizer='L1L2') 
            ]
        )

        self.diagnostician_reg = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(50,)),  # dim(f) = 25
                # tf.keras.layers.Dropout(rate=0.2),  
                # tf.keras.layers.Dense(units=200, activation=tf.nn.leaky_relu),  
                # tf.keras.layers.Dropout(rate=0.2),  
                # tf.keras.layers.Dense(units=100, activation=tf.nn.leaky_relu),  
                tf.keras.layers.Dense(units=1, activation=None, kernel_regularizer='L1L2'),  # 1
            ]
        )

        self.x_attention = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(100,)),
                tf.keras.layers.Dense(units=50, activation='tanh', kernel_regularizer='L1L2'),
                tf.keras.layers.Dense(1, use_bias=False)
            ]
        )
        self.x_attention_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(100,)),
                tf.keras.layers.Dense(units=100, activation='tanh', kernel_regularizer='L1L2'),
            ]
        )

    @tf.function
    def drop_out(self, input, keep_prob):
        return tf.nn.dropout(input, keep_prob)

    # Reconstructed SNPs sampling
    def sample_SNP(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(10, 50))
        return self.decode_SNP(eps, apply_sigmoid=True)

    # Represent mu and sigma from the input SNP
    def encode_SNP(self, x_SNP):
        mean, logvar = tf.split(self.encoder(x_SNP), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize_SNP(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.math.exp(logvar * .5) + mean

    # Reconstruct the input SNP
    def decode_SNP(self, z_SNP, apply_sigmoid=False):
        logits_SNP = self.decoder(z_SNP)
        if apply_sigmoid:
            probs_SNP = tf.math.sigmoid(logits_SNP)
            return probs_SNP
        return logits_SNP

    def encode_MRI(self, x_MRI):
        zb_MRI = self.encoder_MRI(x_MRI)
        return zb_MRI

    def decode_MRI(self, z_MRI):
        xb_MRI = self.decoder_MRI(z_MRI)
        return xb_MRI

    def encode_PET(self, x_PET):
        zb_PET = self.encoder_PET(x_PET)
        return zb_PET

    def decode_PET(self, z_PET):
        xb_PET = self.decoder_PET(z_PET)
        return xb_PET
    
    def self_attention_MRI(self, zb_MRI):
        MRI_ex = tf.expand_dims(zb_MRI, axis=1)
        attention = MultiHeadAttention(num_heads=4, key_dim=50)(MRI_ex, MRI_ex)
        MRI_att = attention[:, 0, :]
        return MRI_att
    
    def self_attention_PET(self, zb_PET):
        PET_ex = tf.expand_dims(zb_PET, axis=1)
        attention = MultiHeadAttention(num_heads=4, key_dim=50)(PET_ex, PET_ex)
        PET_att = attention[:, 0, :]
        return PET_att
    
    def Attention(self, z):
        w = self.x_attention(z)
        beta = tf.nn.softmax(w, axis=1)
        # beta = tf.expand_dims(beta, -1)  # 扩展 beta 的维度以匹配 z
        z_sum = tf.reduce_sum(beta * z, axis=1)
        z_MLP = self.x_attention_mlp(z_sum)
        return z_MLP, beta

    def compute_cosine_similarity(zb_MRI, zb_PET):
        # 计算余弦相似度矩阵
        similarity_matrix = cosine_similarity(zb_MRI, zb_PET)
        return similarity_matrix
    
    def contrastive_loss(similarity_matrix, margin=0.6):
        """
        similarity_matrix: 余弦相似度矩阵
        margin: 对比损失的边际（即不同样本之间的相似度应该低于这个边际）
        """
        num_samples = similarity_matrix.shape[0]
        # 计算对比损失
        loss = 0
        for i in range(num_samples):
            for j in range(num_samples):
                if i == j:
                    # 同一样本的脑区对，目标是让它们更相近
                    loss += np.sum(np.maximum(0, margin - similarity_matrix[i, j]))
                else:
                    # 不同样本的脑区对，目标是让它们更远离
                    loss += np.sum(np.maximum(0, similarity_matrix[i, j] - margin))
        # 归一化损失
        loss /= (num_samples * (num_samples - 1))
        return loss

    # def cross_modal_attention(self, MRI_att, PET_att):
    #     x = tf.expand_dims(MRI_att, axis=1)
    #     y = tf.expand_dims(PET_att, axis=1)
    #     a1 = MultiHeadAttention(num_heads=4, key_dim=50)(x, y)
    #     a2 = MultiHeadAttention(num_heads=4, key_dim=50)(y, x)
    #     MRI_PET = a1[:, 0, :]
    #     PET_MRI = a2[:, 0, :]
    #     # bilinear = BilinearPooling()([x, y])
    #     # bilinear = tf.squeeze(bilinear, axis=1)
    #     return MRI_PET, PET_MRI

    def bilinear_pooling(self, MRI, PET):
        # 假设这里是双线性池化的具体实现
        joint_feature = tf.matmul(MRI, tf.transpose(PET))
        return joint_feature

    def sum_pooling(self, joint_feature):
        compact_feature = tf.reduce_sum(joint_feature, axis=1)
        return compact_feature

    # Attentive vector and fake neuroimaging generation
    def asso_SNP_MRI(self, z_SNP, c_demo):
        z = tf.concat([c_demo, z_SNP], axis=-1)
        x_MRI_fake, x_PET_fake, mask = tf.split(self.asso_MRI(z), num_or_size_splits=[116, 116, 100], axis=1) #此处有问题
        return x_MRI_fake, x_PET_fake, mask
    
    # Classify the real and the fake neuroimaging
    def discriminate_MRI(self, x_MRI_real_or_fake):
        return self.discriminator_MRI(x_MRI_real_or_fake)
    def discriminate_PET(self, x_PET_real_or_fake):
        return self.discriminator_PET(x_PET_real_or_fake)


    def diagnose(self, x_MRI, mask, apply_logistic_activation=False):
        feature = self.diagnostician_share(tf.multiply(x_MRI, mask)) # Hadamard production of the attentive vector
        logit_clf = self.diagnostician_clf(feature)
        logit_reg = self.diagnostician_reg(feature)
        if apply_logistic_activation:
            y_hat = tf.math.softmax(logit_clf)
            s_hat = tf.math.sigmoid(logit_reg)
            return y_hat, s_hat
        return logit_clf, logit_reg

