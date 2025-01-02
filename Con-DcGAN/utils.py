import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler


# Load data samples
def load_dataset(fold, task, SNP_mapping=True, return_tensor=False):
    mytask = []
    is_MCI = False
    for t in task:
        if t == 'CN':
            mytask.append(1)

        if t == 'SMC':
            mytask.append(2)

        if t == 'EMCI':
            mytask.append(3)

        if t == 'LMCI':
            mytask.append(4)

        if t == 'AD':
            mytask.append(5)

        if t == 'MCI': # append sMCI and pMCI samples
            mytask.append(3)
            mytask.append(4)
            is_MCI = True

    mytask = np.array(mytask)

    # Define the data path
    path = 'D:/GAN/Deep learning/Data/'

    Y_dis = np.load(path + 'label.npy')  # Disease labels, CN: 0, sMCI: 1, pMCI: 2, AD: 3

    # Extract appropriate indicies for the given task
    task_idx = np.zeros(shape=Y_dis.shape)

    for t in range(len(mytask)):
        task_idx += np.array(Y_dis == mytask[t])  
    task_idx = task_idx.astype(bool)

    Y_dis = Y_dis[task_idx]

    if is_MCI:
        Y_dis[Y_dis == 4] = 3
    task_idx = np.squeeze(task_idx, 1)
    X_MRI = np.load(path + 'MRIdata.npy')[task_idx, :]  # MRI volume features (N, 116) find the row related to the label
    X_PET = np.load(path + 'FDGdata.npy')[task_idx, :]
    X_SNP = np.load(path + 'SNP.npy')[task_idx, :]  # SNP features (N, 2703)
    C_sex = np.load(path + 'C_sex.npy')[task_idx]  # Sex codes 'F', 'M' (N, )
    C_edu = np.load(path + 'C_edu.npy')[task_idx]  # Education codes (N, )
    C_age = np.load(path + 'C_age.npy')[task_idx]  # Age codes (N, )
    S_cog = np.load(path + 'Score.npy')[task_idx]  # Cognitive scores (MMSE), (N, )

    # One-hot encoding for the disease label  
    for i in range(np.unique(Y_dis).shape[0]):
        Y_dis[Y_dis == np.unique(Y_dis)[i]] = i

    Y_dis = np.eye(np.unique(Y_dis).shape[0])[Y_dis]

    # Normalization
    C_age = C_age / 100
    C_edu = C_edu / 20
    S_cog = S_cog / 30

    # Categorical encoding for the sex code
    for i in range(np.unique(C_sex).shape[0]):
        C_sex[C_sex == np.unique(C_sex)[i]] = i
    #     C_sex = C_sex.astype(np.int)
    #     C_sex = np.eye(np.unique(C_sex).shape[0])[C_sex]

    # Demographic information concatenation
    C_dmg = np.concatenate((C_sex, C_age, C_edu), 1)  # (618, 3)

    # Data randomizing
    rand_idx = np.random.RandomState(seed=951014).permutation(Y_dis.shape[0])  # 使用相同的seed值可以确保在不同的运行中获得相同的结果
    X_MRI = X_MRI[rand_idx, ...]
    X_PET = X_PET[rand_idx, ...]
    X_SNP = X_SNP[rand_idx, ...]
    C_dmg = C_dmg[rand_idx, ...]
    Y_dis = Y_dis[rand_idx, ...]
    S_cog = S_cog[rand_idx, ...]

    # Fold dividing
    rand_idx = np.random.RandomState(seed=5930).permutation(Y_dis.shape[0])
    num_samples = int(Y_dis.shape[0] / 5)
    ts_idx = rand_idx[num_samples * (fold - 1):num_samples * fold]  # 取样本数 0：74，74：148 ...左闭右开
    tr_idx = np.setdiff1d(rand_idx, ts_idx)  # 找到2个数组中集合元素的差异。

    X_MRI_tr, X_MRI_ts = X_MRI[tr_idx, :], X_MRI[ts_idx, :]
    X_PET_tr, X_PET_ts = X_PET[tr_idx, :], X_PET[ts_idx, :]
    X_SNP_tr, X_SNP_ts = X_SNP[tr_idx, :], X_SNP[ts_idx, :]
    C_dmg_tr, C_dmg_ts = C_dmg[tr_idx, :], C_dmg[ts_idx, :]
    Y_dis_tr, Y_dis_ts = Y_dis[tr_idx, :], Y_dis[ts_idx, :]
    S_cog_tr, S_cog_ts = S_cog[tr_idx], S_cog[ts_idx]

    # MRI normalization
    scaler = MinMaxScaler()   
    X_MRI_tr = scaler.fit_transform(X_MRI_tr)
    X_MRI_ts = scaler.transform(X_MRI_ts)
    X_PET_tr = scaler.fit_transform(X_PET_tr)
    X_PET_ts = scaler.transform(X_PET_ts)
    X_SNP_tr = scaler.fit_transform(X_SNP_tr)
    X_SNP_ts = scaler.transform(X_SNP_ts)
    C_dmg_tr = scaler.fit_transform(C_dmg_tr)
    C_dmg_ts = scaler.transform(C_dmg_ts)
    S_cog_tr = scaler.fit_transform(S_cog_tr)
    S_cog_ts = scaler.transform(S_cog_ts)

    if SNP_mapping:
        # SNP encoding
        X_SNP_tr, X_SNP_ts = SNP_encoder(X_SNP_tr=X_SNP_tr, X_SNP_ts=X_SNP_ts)

    if return_tensor:
        return tf.data.Dataset.from_tensor_slices((X_MRI_tr, X_PET_tr, X_SNP_tr, C_dmg_tr, Y_dis_tr, S_cog_tr)), \
               tf.data.Dataset.from_tensor_slices((X_MRI_ts, X_PET_ts, X_SNP_ts, C_dmg_ts, Y_dis_ts, S_cog_ts))

    return X_MRI_tr, X_PET_tr, X_SNP_tr, C_dmg_tr, Y_dis_tr, S_cog_tr, X_MRI_ts, X_PET_ts, X_SNP_ts, C_dmg_ts, Y_dis_ts, S_cog_ts


def SNP_encoder(X_SNP_tr, X_SNP_ts):
    # Based on population, this encoder transforms the discrete SNP vectors to be numerical.
    # The encoder is fit by the training SNP data and applied to the testing SNP data.

    # Fit the encoding table
    encoder = np.empty(shape=(3, X_SNP_tr.shape[1]))
    for i in range(X_SNP_tr.shape[1]):
        for j in [0, 1, 2]:
            encoder[j, i] = np.array(X_SNP_tr[:, i] == j).sum()  # 将每一列的0 1 2的个数对应在0 1 2行

    encoder /= X_SNP_tr.shape[0]  # (3, 1275)

    X_E_SNP_tr = np.empty(shape=X_SNP_tr.shape)
    X_E_SNP_ts = np.empty(shape=X_SNP_ts.shape)

    # Map the SNP values
    for sbj in range(X_SNP_tr.shape[0]):
        for dna in range(X_SNP_tr.shape[-1]):
            X_E_SNP_tr[sbj, dna] = encoder[..., dna][int(X_SNP_tr[sbj, dna])]

    for sbj in range(X_SNP_ts.shape[0]):
        for dna in range(X_SNP_ts.shape[-1]):
            X_E_SNP_ts[sbj, dna] = encoder[..., dna][int(X_SNP_ts[sbj, dna])]

    return X_E_SNP_tr, X_E_SNP_ts


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    pdf = tf.math.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.math.exp(-logvar) + logvar + log2pi), axis=raxis)
    return pdf
