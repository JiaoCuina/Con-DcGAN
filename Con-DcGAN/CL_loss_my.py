import tensorflow as tf

class SupConLoss(tf.keras.layers.Layer):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def call(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = tf.reshape(features, [features.shape[0], features.shape[1], -1])

        batch_size = tf.shape(features)[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = tf.eye(batch_size)
        elif labels is not None:
            labels = tf.reshape(labels, [-1, 1])
            if tf.shape(labels)[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)
        else:
            mask = tf.cast(mask, tf.float32)

        contrast_count = features.shape[1]
        contrast_feature = tf.reshape(tf.transpose(features, [1, 0, 2]), [contrast_count * batch_size, -1])
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = tf.matmul(anchor_feature, contrast_feature, transpose_b=True) / self.temperature
        logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
        logits = anchor_dot_contrast - logits_max

        # tile mask
        mask = tf.tile(mask, [anchor_count, contrast_count])
        # mask-out self-contrast cases
        logits_mask = tf.ones_like(mask) - tf.eye(batch_size * anchor_count)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = tf.exp(logits) * logits_mask
        log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / tf.reduce_sum(mask, axis=1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = tf.reshape(loss, [anchor_count, batch_size])
        loss = tf.reduce_mean(loss)

        return loss
