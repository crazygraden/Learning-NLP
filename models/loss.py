import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class NTXentLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_similarity_function
        else:
            return self._dot_similarity

    def _get_correlated_mask_my(self, time_steps):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy(diag + l1 + l2)
        mask = (1 - mask).type(torch.bool)
        mask = mask.unsqueeze(0).expand(time_steps, -1, -1)
        return mask.to(self.device)

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)  # 对角线元素向下移动k个单位
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)  # 对角线元素向上移动k个单位
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_similarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _cosine_similarity_function(self, x, y):
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v
    # ==== my =====
    def forward(self, zis, zjs):
        time_steps = zis.size(1)
        mask = self._get_correlated_mask_my(time_steps)

        representations = torch.cat([zjs, zis], dim=0)  # 2B x T x C
        similarity_matrix = self.similarity_function(representations, representations)  # 2B x 2B x T
        similarity_matrix = similarity_matrix.permute(2, 0, 1)  # T x 2B x 2B

        l_pos = torch.stack([similarity_matrix[i].diagonal(self.batch_size) for i in range(time_steps)])  # T x B
        r_pos = torch.stack([similarity_matrix[i].diagonal(-self.batch_size) for i in range(time_steps)])  # T x B
        positives = torch.cat([l_pos, r_pos], dim=1).unsqueeze(-1)  # T x 2B x 1
        negatives = similarity_matrix[mask].view(time_steps, 2 * self.batch_size, -1)  # T x 2B x (2B-2)
        logits = torch.cat((positives, negatives), dim=2)  # T x 2B x (2B-1)
        logits /= self.temperature

        labels = torch.zeros(time_steps, 2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss / (2 * self.batch_size * time_steps)

    # # ======================= orig =========================================
    # def forward(self, zis, zjs):
    #     representations = torch.cat([zjs, zis], dim=0)
    #     similarity_matrix = self.similarity_function(representations, representations)
    #
    #     # filter out the scores from the positive samples
    #     l_pos = torch.diag(similarity_matrix, self.batch_size)
    #     r_pos = torch.diag(similarity_matrix, -self.batch_size)
    #     positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
    #
    #     negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
    #
    #     logits = torch.cat((positives, negatives), dim=1)  # 沿着 列 拼接张量序列
    #     logits /= self.temperature
    #
    #     labels = torch.zeros(2 * self.batch_size).to(self.device).long()
    #     loss = self.criterion(logits, labels)
    #
    #     return loss / (2 * self.batch_size)



def instance_contrastive_loss(z1, z2):
    # print(z1.size(), z2.size())  # torch.Size([32, 512, 200])
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    # print(z.size())   # torch.Size([64, 512, 200])
    z = z.transpose(0, 1)  # T x 2B x C
    # print(z.size())  # torch.Size([512, 64, 200])
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    # print(sim.size())   # torch.Size([512, 64, 64])
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


class LDAMLoss(torch.nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=20):
        super(LDAMLoss, self).__init__()
        cls_num_list = np.array(cls_num_list, dtype=np.float32)
        cls_num_list[cls_num_list == 0] = 1e-5
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        if np.max(m_list) == 0:
            # 处理 np.max(m_list) 为零的情况
            print("Warning: np.max(m_list) is zero. Adjusting the computation to avoid division by zero.")
            # 根据需要调整，比如设置一个非常小的值，避免除以零
            m_list = m_list * (max_m / (np.max(m_list) + 1e-10))
        else:
            m_list = m_list * (max_m / np.max(m_list))
        # m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor)
        # print(self.m_list[None, :])
        # print(index_float.transpose(0, 1))

        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)



class ClassBalancedLoss:
    def __init__(self, samples_per_cls, no_of_classes, beta=0.99, gamma=0.9, loss_type="cross_entropy"):
        """
        Initialize the Class Balanced Loss.

        Args:
          samples_per_cls: A list of size [no_of_classes] indicating the number of samples for each class.
          no_of_classes: Total number of classes. int
          beta: Hyperparameter for Class Balanced Loss.
          gamma: Hyperparameter for Focal Loss.
          loss_type: string. One of "sigmoid", "focal", "softmax".
        """
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.weights = self._calculate_class_weights()
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.weights)

    def _calculate_class_weights(self):
        """Calculate class weights based on the effective number of samples."""
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        effective_num = np.clip(effective_num, a_min=1e-10, a_max=None)  # 避免出现 0
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.no_of_classes
        return torch.tensor(weights).float()

    def focal_loss(self, labels, logits, alpha):
        """Compute the focal loss between `logits` and the ground truth `labels`."""
        BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels * logits - self.gamma * torch.log(1 + torch.exp(-1.0 * logits)))
        loss = modulator * BCLoss
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
        focal_loss /= torch.sum(labels)
        return focal_loss

    def compute(self, labels, logits):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

        Args:
          labels: A tensor of size [batch].
          logits: A tensor of size [batch, no_of_classes].

        Returns:
          cb_loss: A tensor representing class balanced loss.
        """
        labels_one_hot = F.one_hot(labels, self.no_of_classes).float()

        weights = self.weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1).unsqueeze(1).repeat(1, self.no_of_classes)
        if self.loss_type == "focal":
            cb_loss = self.focal_loss(labels_one_hot, logits, weights)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        elif self.loss_type == "cross_entropy":
            cb_loss = self.cross_entropy_loss(logits, labels)  # Use CrossEntropyLoss directly
        return cb_loss