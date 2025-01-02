# import APIs
import utils
import engine
import ETest
import datetime
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import matplotlib.pyplot as plt
# import os
from sklearn.preprocessing import label_binarize
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve, auc, accuracy_score, mean_squared_error, confusion_matrix
from CL_loss_my import SupConLoss

config = tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

## 一个生成器，两个鉴别器，engine  WWRIGHT

class experiment():
    def __init__(self, fold_idx, task):
        self.fold_idx = fold_idx
        self.task = task

        # Learning schedules
        self.num_epochs = 200 # 100
        self.num_batches = 5
        self.lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=1000,
                                                                 decay_rate=.96, staircase=False)  # init_lr: 1e-3

        # Loss control hyperparameter
        self.alpha_rec_SNP = .7  # reconstruction
        self.alpha_rec = .7
        self.alpha_cl_image = .7
        self.alpha_gen = .5 # generation
        self.alpha_dis = 1  # discrimination
        self.alpha_clf = 1  # classification
        self.alpha_reg = 0  # regression
        
    def training(self):
        print(f'Start Training, Fold {self.fold_idx}')

        # Load dataset
        X_MRI_train, X_PET_train, E_SNP_train, C_demo_train, Y_train, S_train, \
        X_MRI_test, X_PET_test, E_SNP_test, C_demo_test, Y_test, S_test = utils.load_dataset(self.fold_idx, self.task)
        N_o = Y_train.shape[-1]
        ## 数据重采样
        # class_weights_MRI = calculate_class_weights(Y_train)
        # resampling_indices_MRI = get_resampling_indices(Y_train, class_weights_MRI)
        # class_weights_PET = calculate_class_weights(Y_train)
        # resampling_indices_PET = get_resampling_indices(Y_train, class_weights_PET)
        # class_weights_SNP = calculate_class_weights(Y_train)
        # resampling_indices_SNP = get_resampling_indices(Y_train, class_weights_SNP)
        # class_weights_demo = calculate_class_weights(Y_train)
        # resampling_indices_demo = get_resampling_indices(Y_train, class_weights_demo)
        # class_weights_S = calculate_class_weights(Y_train)
        # resampling_indices_S = get_resampling_indices(Y_train, class_weights_S)

        # X_MRI_train_resampled = X_MRI_train[resampling_indices_MRI, ...]
        # Y_train_resampled = Y_train[resampling_indices_MRI, ...]
        # X_PET_train_resampled = X_PET_train[resampling_indices_PET, ...]
        # E_SNP_train_resampled = E_SNP_train[resampling_indices_SNP, ...]
        # C_demo_train_resampled = C_demo_train[resampling_indices_demo, ...]
        # S_train_resampled = S_train[resampling_indices_S, ...]

        # Call optimizers
        opt_rec_SNP = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_rec_MRI = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_rec_PET = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_gen = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_dis_mri = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_dis_pet = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_CON_PET = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_CON_MRI = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_clf = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_reg = tf.keras.optimizers.Adam(learning_rate=self.lr)

        model = engine.engine(N_o=N_o)

        num_iters = int(Y_train.shape[0]/self.num_batches)

        for epoch in range(self.num_epochs):
            L_rec_SNP_per_epoch = 0
            L_rec_MRI_per_epoch = 0
            L_rec_PET_per_epoch = 0
            L_con_MRI_per_epoch = 0
            L_con_PET_per_epoch = 0
            L_gen_per_epoch = 0
            L_dis_per_epoch = 0
            L_clf_per_epoch = 0
            L_reg_per_epoch = 0

            # np.random.seed(42)
            rand_idx = np.random.permutation(Y_train.shape[0])
            # rand_idx = np.random.permutation(len(Y_train_resampled))
            X_MRI_train = X_MRI_train[rand_idx, ...]
            X_PET_train = X_PET_train[rand_idx, ...]
            E_SNP_train = E_SNP_train[rand_idx, ...]
            C_demo_train = C_demo_train[rand_idx, ...]
            Y_train = Y_train[rand_idx, ...]
            S_train = S_train[rand_idx, ...]

            for batch in range(num_iters):
                # Sample a minibatch
                xb_MRI = X_MRI_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...].astype(np.float32)
                xb_PET = X_PET_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...].astype(np.float32)
                eb_SNP = E_SNP_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...].astype(np.float32)
                cb_demo = C_demo_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...]
                yb_clf = Y_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...].astype(np.float32)
                sb_reg = S_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...]

                # Estimate gradient and loss
                with tf.GradientTape() as tape_rec_SNP, tf.GradientTape() as tape_rec_MRI, tf.GradientTape() as tape_rec_PET, \
                     tf.GradientTape() as tape_CON_MRI, tf.GradientTape() as tape_CON_PET, \
                    tf.GradientTape() as tape_gen, tf.GradientTape() as tape_dis_mri, tf.GradientTape() as tape_dis_pet, tf.GradientTape() as tape_clf, tf.GradientTape() as tape_reg:

                    # SNP representation module
                    mu, log_sigma_square = model.encode_SNP(x_SNP=eb_SNP)
                    zb_SNP = model.reparameterize_SNP(mean=mu, logvar=log_sigma_square)
                    eb_SNP_hat_logit = model.decode_SNP(z_SNP=zb_SNP)
                    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=eb_SNP_hat_logit, labels=eb_SNP)
                    log_prob_eb_SNP_given_zb_SNP = -tf.math.reduce_sum(cross_ent, axis=1)
                    log_prob_zb_SNP = utils.log_normal_pdf(sample=zb_SNP, mean=0., logvar=0.)
                    log_q_zb_given_eb_SNP = utils.log_normal_pdf(sample=zb_SNP, mean=mu, logvar=log_sigma_square)

                    # SNP Reconstruction loss SNP重构损失
                    L_rec_SNP = -tf.math.reduce_mean(log_prob_eb_SNP_given_zb_SNP + log_prob_zb_SNP - log_q_zb_given_eb_SNP)
                    L_rec_SNP *= self.alpha_rec_SNP

                    # Imaging representation module
                    zb_MRI = model.encode_MRI(x_MRI=xb_MRI)  # 编码器得到潜在空间样本
                    zb_PET = model.encode_PET(x_PET=xb_PET)  # 编码器得到潜在空间样本
                    
                    eb_MRI = model.decode_MRI(z_MRI=zb_MRI)
                    eb_PET = model.decode_PET(z_PET=zb_PET)
                    # L_rec_MRI = tf.reduce_mean(tf.square(eb_MRI - xb_MRI))
                    L_rec_MRI = tf.reduce_mean(
                        tf.sqrt(tf.keras.losses.mean_squared_error(eb_MRI, xb_MRI)))
                    L_rec_MRI *= self.alpha_rec
                    L_rec_PET = tf.reduce_mean(
                        tf.sqrt(tf.keras.losses.mean_squared_error(eb_PET, xb_PET)))
                    # L_rec_PET = tf.reduce_mean(tf.square(eb_PET - xb_PET))
                    L_rec_PET *= self.alpha_rec
                    

                    MRI_l2_norm = tf.nn.l2_normalize(zb_MRI, axis=1)
                    PET_l2_norm = tf.nn.l2_normalize(zb_PET, axis=1)

                    MRI_l2_norm_expanded = tf.expand_dims(MRI_l2_norm, axis=1)  # 形状变为 [5, 1, 200]
                    PET_l2_norm_expanded = tf.expand_dims(PET_l2_norm, axis=1)  # 形状变为 [5, 1, 200]

                    features_cl_image = tf.concat([MRI_l2_norm_expanded, PET_l2_norm_expanded], axis=1)
                    
                    criterion_MRI_PET_image = SupConLoss(temperature=0.1, base_temperature=0.1)
                    L_CL_image = criterion_MRI_PET_image.call(features_cl_image, labels=None, mask=None)
                    L_CL_image *= self.alpha_cl_image

                    ## 注意力加权多模态数据融合
                    # joint_feature = tf.concat([zb_MRI, zb_PET], axis=-1)
                    joint_feature = tf.stack([zb_MRI, zb_PET], axis=1)
                    joint_feature,att = model.Attention(joint_feature)

                    zb_fake_mri, zb_fake_pet, mask_a = model.asso_SNP_MRI(z_SNP=zb_SNP, c_demo=cb_demo) # 关联得到的假的

                    real_output_mri = model.discriminate_MRI(x_MRI_real_or_fake=xb_MRI)
                    fake_output_mri = model.discriminate_MRI(x_MRI_real_or_fake=zb_fake_mri)

                    real_output_pet = model.discriminate_PET(x_PET_real_or_fake=xb_PET)
                    fake_output_pet = model.discriminate_PET(x_PET_real_or_fake=zb_fake_pet)
                    ## Least-Square GAN loss
                    L_gen_mri = tf.keras.losses.MSE(tf.ones_like(fake_output_mri), fake_output_mri)
                    L_gen_pet = tf.keras.losses.MSE(tf.ones_like(fake_output_pet), fake_output_pet)

                    L_gen = (L_gen_mri + L_gen_pet) / 2
                    L_gen *= self.alpha_gen 
                    
                    ## New GAN loss
                    # L_gen_mri = tf.reduce_mean(
                    #     tf.sqrt(tf.keras.losses.mean_squared_error(xb_MRI, zb_fake_mri)))
                    # L_gen_pet = tf.reduce_mean(
                    #     tf.sqrt(tf.keras.losses.mean_squared_error(xb_PET, zb_fake_pet)))
                    # L_gen = (L_gen_mri + L_gen_pet) / 2
                    # L_gen *= self.alpha_gen

                    L_dis_mri = tf.keras.losses.MSE(tf.ones_like(real_output_mri), real_output_mri) \
                            + tf.keras.losses.MSE(tf.zeros_like(fake_output_mri), fake_output_mri)
                    # L_dis_mri *= self.alpha_dis
                    L_dis_pet = tf.keras.losses.MSE(tf.ones_like(real_output_pet), real_output_pet) \
                            + tf.keras.losses.MSE(tf.zeros_like(fake_output_pet), fake_output_pet)
                    # L_dis_pet *= self.alpha_dis2
                    L_dis = (L_dis_mri + L_dis_pet)/2
                    L_dis *= self.alpha_dis
                    
                    # Diagnostician module
                    yb_clf_hat, sb_reg_hat = model.diagnose(x_MRI=joint_feature, mask=mask_a, apply_logistic_activation=True)

                    # Classification loss 分类损失
                    L_clf = tfa.losses.sigmoid_focal_crossentropy(yb_clf, yb_clf_hat)
                    L_clf *= self.alpha_clf

                    # Regression loss 回归损失
                    L_reg = tf.keras.losses.MSE(sb_reg, sb_reg_hat)
                    L_reg *= self.alpha_reg
                
                # 获取模型中所有可训练的变量
                var = model.trainable_variables

                # 将可训练的变量按模块分组，方便后续的梯度计算和优化
                # 这些变量通常包括权重和偏置
                theta_SNP_Encoder = [var[0], var[1], var[2], var[3]]
                theta_SNP_Decoder = [var[4], var[5], var[6], var[7]]
                theta_MRI_Encoder = [var[8], var[9], var[10], var[11]]
                theta_MRI_Decoder = [var[12], var[13], var[14], var[15]]
                theta_PET_Encoder = [var[16], var[17], var[18], var[19]]
                theta_PET_Decoder = [var[20], var[21], var[22], var[23]]
                theta_gen = [var[24], var[25], var[26], var[27]]
                theta_D_MRI = [var[28], var[29], var[30], var[31]] 
                theta_D_PET = [var[32], var[33], var[34], var[35]]              
                theta_C_share = [var[36], var[37]]
                theta_C_clf = [var[38], var[39]]
                theta_C_reg = [var[40], var[41]]
                theta_C_att = [var[42], var[43], var[44], var[45], var[46]]
                   

                # 计算损失函数 L_rec_SNP 对 SNP 编码器和解码器参数的梯度
                grad_rec_SNP = tape_rec_SNP.gradient(L_rec_SNP, theta_SNP_Encoder + theta_SNP_Decoder)
                # 使用优化器更新这些参数
                opt_rec_SNP.apply_gradients(zip(grad_rec_SNP, theta_SNP_Encoder + theta_SNP_Decoder))
                # 累加每个 epoch 的损失，用于后续的平均计算
                L_rec_SNP_per_epoch += np.mean(L_rec_SNP)

                grad_rec_MRI = tape_rec_MRI.gradient(L_rec_MRI, theta_MRI_Encoder + theta_MRI_Decoder)
                opt_rec_MRI.apply_gradients(zip(grad_rec_MRI, theta_MRI_Encoder + theta_MRI_Decoder))
                L_rec_MRI_per_epoch += np.mean(L_rec_MRI)

                grad_rec_PET = tape_rec_PET.gradient(L_rec_PET, theta_PET_Encoder + theta_PET_Decoder)
                opt_rec_PET.apply_gradients(zip(grad_rec_PET, theta_PET_Encoder + theta_PET_Decoder))
                L_rec_PET_per_epoch += np.mean(L_rec_PET)
                #
                # grad_CON_MRI = tape_CON_MRI.gradient(L_CL_image, theta_MRI_Encoder)
                # opt_CON_MRI.apply_gradients(zip(grad_CON_MRI, theta_MRI_Encoder))
                # L_con_MRI_per_epoch += np.mean(L_CL_image)

                # grad_CON_PET = tape_CON_PET.gradient(L_CL_image, theta_PET_Encoder)
                # opt_CON_PET.apply_gradients(zip(grad_CON_PET, theta_PET_Encoder))
                # L_con_PET_per_epoch += np.mean(L_CL_image)

                grad_CON_MRI = tape_CON_MRI.gradient(L_CL_image, theta_MRI_Encoder + theta_PET_Encoder)
                opt_CON_MRI.apply_gradients(zip(grad_CON_MRI, theta_MRI_Encoder + theta_PET_Encoder))
                L_con_MRI_per_epoch += np.mean(L_CL_image)
                #

                # grad_gen = tape_gen.gradient(L_gen, theta_MRI_Encoder + theta_PET_Encoder + theta_SNP_Encoder + theta_gen)
                # opt_gen.apply_gradients(zip(grad_gen, theta_MRI_Encoder + theta_PET_Encoder + theta_SNP_Encoder + theta_gen))
                # L_gen_per_epoch += np.mean(L_gen)

                grad_gen = tape_gen.gradient(L_gen, theta_SNP_Encoder + theta_gen)
                opt_gen.apply_gradients(zip(grad_gen, theta_SNP_Encoder + theta_gen))
                L_gen_per_epoch += np.mean(L_gen)

                grad_dis_MRI = tape_dis_mri.gradient(L_dis_mri, theta_D_MRI)
                opt_dis_mri.apply_gradients(zip(grad_dis_MRI, theta_D_MRI))
                grad_dis_PET = tape_dis_pet.gradient(L_dis_pet, theta_D_PET)
                opt_dis_pet.apply_gradients(zip(grad_dis_PET, theta_D_PET))
                L_dis_per_epoch += np.mean(L_dis)
                # L_dis_mri_per_epoch += np.mean(L_dis_mri)
                # L_dis_pet_per_epoch += np.mean(L_dis_pet)

                grad_clf = tape_clf.gradient(L_clf, theta_MRI_Encoder + theta_PET_Encoder + theta_gen + theta_C_share + theta_C_clf + theta_C_att)
                opt_clf.apply_gradients(zip(grad_clf, theta_MRI_Encoder + theta_PET_Encoder + theta_gen + theta_C_share + theta_C_clf + theta_C_att))
                L_clf_per_epoch += np.mean(L_clf)

                grad_reg = tape_reg.gradient(L_reg, theta_MRI_Encoder + theta_PET_Encoder + theta_gen + theta_C_share + theta_C_reg + theta_C_att)
                opt_reg.apply_gradients(zip(grad_reg, theta_MRI_Encoder + theta_PET_Encoder + theta_gen + theta_C_share + theta_C_reg + theta_C_att))
                L_reg_per_epoch += np.mean(L_reg)

            L_rec_SNP_per_epoch /= num_iters
            L_rec_MRI_per_epoch /= num_iters
            L_rec_PET_per_epoch /= num_iters
            L_con_MRI_per_epoch /= num_iters
            L_con_PET_per_epoch /= num_iters
            L_gen_per_epoch /= num_iters
            L_dis_per_epoch /= num_iters
            # L_dis_pet_per_epoch /= num_iters
            L_clf_per_epoch /= num_iters
            L_reg_per_epoch /= num_iters

            # Loss report
            print(f'Epoch: {epoch + 1}, Lrec_SNP: {L_rec_SNP_per_epoch:>.4f}, Lrec_MRI: {L_rec_MRI_per_epoch:>.4f}, Lrec_PET: {L_rec_PET_per_epoch:>.4f}, L_gen: {L_gen_per_epoch:>.4f}, L_CON: {L_con_MRI_per_epoch:>.4f}, L_dis_MRI: {L_dis_per_epoch:>.4f}, Lclf: {L_clf_per_epoch:>.4f}')

        # Results
        mu_SNP, log_sigma_square_SNP = model.encode_SNP(E_SNP_test)
        Z_SNP_test = model.reparameterize_SNP(mu_SNP, log_sigma_square_SNP)
        MRI = model.encode_MRI(X_MRI_test)
        PET = model.encode_PET(X_PET_test)
        joint_feature = tf.stack([MRI, PET], axis=1)
        joint_feature,att = model.Attention(joint_feature)
        _, _, A_test = model.asso_SNP_MRI(Z_SNP_test, C_demo_test)
        Y_test_hat, S_test_hat = model.diagnose(joint_feature, A_test, True)
        return Y_test, S_test, Y_test_hat, S_test_hat
    
def calculate_class_weights(Y):
    from sklearn.utils.class_weight import compute_class_weight
    class_labels = np.argmax(Y, axis=1)
    classes = np.unique(class_labels)
    class_weight = compute_class_weight('balanced', classes=classes, y=class_labels)
    return dict(zip(classes, class_weight))

def get_resampling_indices(Y, class_weights):
    indices = []
    class_labels = np.argmax(Y, axis=1)
    for cls, weight in class_weights.items():
        class_indices = np.where(class_labels == cls)[0]
        if weight > 0:
            selected_indices = np.random.choice(class_indices, size=int(len(class_indices) * weight), replace=True)
            indices.extend(selected_indices)
    return indices

def calculate_accuracy(y_pred, y_true):
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    correct_predictions = np.sum(y_pred_labels == y_true_labels)
    accuracy = correct_predictions / len(y_true_labels)
    return accuracy

def calculate_sen_spe(y_pred, y_true):
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    tp = np.sum((y_pred_labels == 1) & (y_true_labels == 1))
    fp = np.sum((y_pred_labels == 1) & (y_true_labels == 0))
    tn = np.sum((y_pred_labels == 0) & (y_true_labels == 0))
    fn = np.sum((y_pred_labels == 0) & (y_true_labels == 1))
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    return sen, spe

def calculate_mcc(y_pred, y_true):
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    tp = np.sum((y_pred_labels == 1) & (y_true_labels == 1))
    tn = np.sum((y_pred_labels == 0) & (y_true_labels == 0))
    fp = np.sum((y_pred_labels == 1) & (y_true_labels == 0))
    fn = np.sum((y_pred_labels == 0) & (y_true_labels == 1))
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return mcc

task = ['CN', 'AD'] 
# ['CN', 'MCI'], ['sMCI', 'pMCI'], ['CN', 'MCI', 'AD'], ['CN', 'sMCI', 'pMCI', 'AD']
auc_list = []
acc_list = []
sen_list = []
spe_list = []
f1_list = []
BAC_list = []
pre_list = []
rmse_list = []
tprs = []
tpr_list=[]
fpr_list=[]
all_roc_auc=[]
mean_fpr=np.linspace(0,1,100)
fold_metrics_list = []
metrics_list = []
fig, ax = plt.subplots()
colors = ['red', 'blue', 'green', 'yellow', 'black']
for fold in range(5):  # five-fold cross-validation
    exp = experiment(fold + 1, ['CN', 'AD'])
    Y_test, S_test, Y_test_hat, S_test_hat = exp.training()
    y_pred_labels = np.argmax(Y_test_hat, axis=1)
    y_true_labels = np.argmax(Y_test, axis=1)
    fpr, tpr, _ = roc_curve(Y_test[:,1], Y_test_hat[:,1])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[fold], label='Fold %d (AUC = %0.4f)' % (fold + 1, roc_auc))
    fpr_list.append(fpr)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    # np.savetxt(f'fpr_CNAD_{fold}.csv', fpr, delimiter=',')
    # np.savetxt(f'tpr_CNAD_{fold}.csv', tpr, delimiter=',')
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    np.savetxt(f'D:/ResearchGroup/GAN/Biomarkers/1_{fold}_{timestamp}.csv', np.column_stack((fpr, tpr)), delimiter=',', fmt='%.6f', header='FPR,TPR')
    AUC = roc_auc_score(Y_test, Y_test_hat)
    print(f'Test AUC: {AUC:>.4f}')
    accuracy = calculate_accuracy(Y_test_hat, Y_test)
    print(f'Test ACC: {accuracy:>.4f}')
    sen, spe = calculate_sen_spe(Y_test_hat, Y_test)
    print(f'Test SPE: {spe:>.4f}')
    precision = precision_score(y_true_labels, y_pred_labels)
    print(f'Test Pre: {precision:>.4f}')
    bac = balanced_accuracy_score(y_true_labels, y_pred_labels)
    print(f'Test BAC: {bac:>.4f}')
    rmse = np.sqrt(mean_squared_error(S_test * 30., S_test_hat * 30.))
    print(f'Test Regression RMSE: {rmse:>.4f}')
    auc_list.append(AUC)
    acc_list.append(accuracy)
    spe_list.append(spe)
    BAC_list.append(bac)
    pre_list.append(precision)
    rmse_list.append(rmse)
    all_roc_auc.append(roc_auc)
mean_auc = np.mean(auc_list)
mean_acc = np.mean(acc_list)
mean_spe = np.mean(spe_list)
mean_BAC = np.mean(BAC_list)
mean_rmse = np.mean(rmse_list)
mean_pre = np.mean(pre_list)
mean_roc = np.mean(all_roc_auc)

# interpolated_tprs = []
# for i in range(len(fpr_list)):
#     # 对每个 fpr 进行插值，tpr_list[i] 是与 fpr_list[i] 对应的 tpr
#     tpr_interp = np.interp(mean_fpr, fpr_list[i], tpr_list[i])
#     interpolated_tprs.append(tpr_interp)
# mean_tpr = np.mean(interpolated_tprs, axis=0)

mean_roc_auc = np.mean(all_roc_auc)
mean_tpr=np.mean(tprs,axis=0)
mean_tpr[-1]=1.0

std_auc = np.std(auc_list)
std_acc = np.std(acc_list)
std_spe = np.std(spe_list)
std_BAC = np.std(BAC_list)
std_rmse = np.std(rmse_list)
std_pre = np.std(pre_list)
std_roc = np.std(all_roc_auc)

print(f'Mean AUC: {mean_auc:>.4f} '
      f'Mean ACC: {mean_acc:>.4f} '
      f'Mean SPE: {mean_spe:>.4f} '
      f'Mean PRE: {mean_pre:>.4f} '
      f'Mean BAC: {mean_BAC:>.4f} '
      f'Mean RMSE: {mean_rmse:>.4f}')

print(f'Std AUC: {std_auc:>.4f} '
      f'Std ACC: {std_acc:>.4f} '
      f'Std SPE: {std_spe:>.4f} '
      f'Std PRE: {std_pre:>.4f} '
      f'Std BAC: {std_BAC:>.4f} '
      f'Std RMSE: {std_rmse:>.4f}')
