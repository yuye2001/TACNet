import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
import pdb


class OriTripletLoss(nn.Module):


    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)


        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()


        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)


        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target,prob1=None,prob2=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        if prob1==None:
            XX = kernels[:batch_size, :batch_size]
            YY = kernels[batch_size:, batch_size:]
            XY = kernels[:batch_size, batch_size:]
            YX = kernels[batch_size:, :batch_size]
        else:
            XX = kernels[:batch_size, :batch_size] * prob1.to('cuda')
            YY = kernels[batch_size:, batch_size:] * prob2.to('cuda')
            XY = kernels[:batch_size, batch_size:] * (prob1.to('cuda') + prob2.to('cuda')) / 2
            YX = kernels[batch_size:, :batch_size] * (prob1.to('cuda') + prob2.to('cuda')) / 2

        loss = torch.mean(XX + YY - XY - YX)
        return loss

def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class TripletLoss_WRT(nn.Module):


    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, label_assign, targets, true_targets, prob, threshold=0.6, alpha=100, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct, closest_negative.shape[0]

class TripletLoss_ADP(nn.Module):
    def __init__(self, alpha=1, gamma=1, square=0):
        super(TripletLoss_ADP, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.square = square

    def forward(self, inputs, label_assign, targets, true_targets, prob, threshold=0.6, alpha=100, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap * self.alpha, is_pos)
        weights_an = softmax_weights(-dist_an * self.alpha, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        if self.square == 0:
            y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
            loss = self.ranking_loss(self.gamma * (closest_negative - furthest_positive), y)
        else:
            diff_pow = torch.pow(furthest_positive - closest_negative, 2) * self.gamma
            diff_pow = torch.clamp_max(diff_pow, max=88)
            y1 = (furthest_positive > closest_negative).float()
            y2 = y1 - 1
            y = -(y1 + y2)
            loss = self.ranking_loss(diff_pow, y)

        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct, closest_negative.shape[0]

class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()

    def forward(self, pred, label):
        T = 3
        predict = F.log_softmax(pred / T, dim=1)
        target_data = F.softmax(label / T, dim=1)
        target_data = target_data + 10 ** (-7)
        target = Variable(target_data.data.cuda(), requires_grad=False)
        loss = T * T * ((target * (target.log() - predict)).sum(1).sum() / target.size()[0])
        return loss

def pdist_torch(emb1, emb2):
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx

def pdist_np(emb1, emb2):
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis=1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis=1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    return dist_mtx

class RobustTripletLoss_final(nn.Module):
    def __init__(self, batch_size, margin):
        super(RobustTripletLoss_final, self).__init__()
        self.batch_size = batch_size
        self.margin = margin
        self.T=0.1

    def forward(self, inputs, prediction, targets, true_targets, prob, threshold):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()

        is_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
        is_neg = targets.expand(n, n).ne(targets.expand(n, n).t())
        is_confident = (prob >= threshold)
        dist_ap, dist_an = [], []
        cnt, loss = 0, 0
        tnt=0
        loss_inverse = False
        K = 20

        for i in range(n):
            if is_confident[i]:
                pos_idx = (torch.nonzero(is_pos[i].long())).squeeze(1)
                neg_idx = (torch.nonzero(is_neg[i].long())).squeeze(1)

                random_pos_index = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                endwhile=0
                while random_pos_index == i:
                    endwhile+=1
                    random_pos_index = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                    if endwhile>10:
                        break

                rank_neg_index = dist[i][neg_idx].argsort()
                hard_neg_index = rank_neg_index[0]
                hard_neg_index = neg_idx[hard_neg_index]

                dist_ap.append(dist[i][random_pos_index].unsqueeze(0))
                dist_an.append(dist[i][hard_neg_index].unsqueeze(0))

                if prob[random_pos_index] >= threshold and prob[hard_neg_index] >= threshold:
                    pass

                elif prob[random_pos_index] >= threshold and prob[hard_neg_index] < threshold:
                    is_FN = (torch.argmax(prediction[hard_neg_index]) == targets[i])
                    if is_FN:
                        tmp = rank_neg_index[1]
                        hard_neg_index_new = neg_idx[tmp]
                        j = 1
                        loop_cnt = 0
                        while prob[hard_neg_index_new] < threshold:
                            j += 1
                            tmp = rank_neg_index[j]
                            hard_neg_index_new = neg_idx[tmp]
                            loop_cnt += 1
                            if loop_cnt >= 10:
                                break
                        dist_ap[cnt] = (dist[i][random_pos_index].unsqueeze(0) + dist[i][hard_neg_index].unsqueeze(0)) / 2
                        dist_an[cnt] = dist[i][hard_neg_index_new].unsqueeze(0)
                    else:
                        pass

                elif prob[random_pos_index] < threshold and prob[hard_neg_index] >= threshold:
                    random_pos_index_new = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                    loop_cnt = 0
                    while random_pos_index_new == i or prob[random_pos_index_new] < threshold:
                        random_pos_index_new = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                        loop_cnt += 1
                        if loop_cnt >= 5:
                            break
                    dist_an[cnt] = (dist[i][random_pos_index].unsqueeze(0)+ dist[i][hard_neg_index].unsqueeze(0)) / 2
                    dist_ap[cnt] = dist[i][random_pos_index_new].unsqueeze(0)

                elif prob[random_pos_index] < threshold and prob[hard_neg_index] < threshold:
                    is_FN = (torch.argmax(prediction[hard_neg_index]) == targets[i])
                    if is_FN:
                        loss_inverse = True
                    else:
                        random_pos_index_new = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                        loop_cnt = 0
                        while random_pos_index_new == i or prob[random_pos_index_new] < threshold:
                            random_pos_index_new = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                            loop_cnt += 1
                            if loop_cnt >= 5:
                                break
                        dist_an[cnt] = (dist[i][random_pos_index].unsqueeze(0)+ dist[i][hard_neg_index].unsqueeze(0)) / 2
                        dist_ap[cnt] = dist[i][random_pos_index_new].unsqueeze(0)

                if loss_inverse:
                    loss += torch.clamp(dist_an[cnt] - dist_ap[cnt] + self.margin, 0)
                else:
                    loss += torch.clamp(dist_ap[cnt] - dist_an[cnt] + self.margin, 0)
                cnt += 1
                tnt+=1
                loss_inverse = False
            else:
                cln=0.01
                if i<=31:
                    ap_dis=dist[i][i+32]
                    V_nagetive=dist[i][0:32]
                    I_nagetive = dist[i][64:]
                    V_an_dis, AV_indices = torch.sort(V_nagetive, dim=0, descending=True)
                    I_an_dis, AI_indices = torch.sort(I_nagetive, dim=0, descending=True)
                    V_an_dis = dist[i][AV_indices[0:K]]
                    I_an_dis = dist[i][AI_indices[0:K]]
                    V_an_dis=torch.sum(V_an_dis, dim=0)/K
                    I_an_dis=torch.sum(I_an_dis, dim=0)/K
                    loss = (torch.clamp(ap_dis - (I_an_dis)+ self.margin, 0)+torch.clamp(ap_dis - (V_an_dis)+ self.margin, 0))/2

                elif i>=32 and i<64:
                    ap_dis = dist[i][i - 32]
                    V_nagetive = dist[i][32:64]
                    I_nagetive = dist[i][64:]
                    V_an_dis, AV_indices = torch.sort(V_nagetive, dim=0, descending=True)
                    I_an_dis, AI_indices = torch.sort(I_nagetive, dim=0, descending=True)
                    V_an_dis = dist[i][AV_indices[0:K]]
                    I_an_dis = dist[i][AI_indices[0:K]]
                    V_an_dis = torch.sum(V_an_dis, dim=0)/K
                    I_an_dis = torch.sum(I_an_dis, dim=0)/K
                    loss = (torch.clamp(ap_dis - (I_an_dis)+ self.margin, 0)+torch.clamp(ap_dis - (V_an_dis)+ self.margin, 0))/2

                else:
                    ap_dis = dist[i][i]
                    V_nagetive = dist[i][64:]
                    an_dis, AV_indices = torch.sort(V_nagetive, dim=0, descending=True)
                    an_dis = dist[i][AV_indices[0:K]]
                    an_dis = torch.sum(an_dis, dim=0)/K
                    loss += torch.clamp(ap_dis - an_dis + self.margin, 0)
                tnt += 1
                loss=loss.reshape(1,-1)

        if cnt == 0:
            return torch.Tensor([0.]).to(inputs.device), 0, cnt
        else:
            dist_ap = torch.cat(dist_ap)
            dist_an = torch.cat(dist_an)
            correct = torch.ge(dist_an, dist_ap).sum().item()
            return loss / tnt, correct, cnt


class DCALLoss(nn.Module):

    def __init__(self, cls_thresh=0.6, consist_thresh=0.45):
        super().__init__()
        self.cls_thresh = cls_thresh
        self.consist_thresh = consist_thresh
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, pseudo_pids, valid_mask, adaptive_w):
        loss = self.ce(logits, pseudo_pids)
        loss = loss * valid_mask.float() * adaptive_w
        return loss.mean()

class GLPRLoss(nn.Module):

    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp

    def forward(self, refined_feat, raw_feat):
        sim = F.cosine_similarity(refined_feat, raw_feat, dim=-1)
        loss = 1 - sim.mean()
        return loss

class CCMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, vis_cal, ir_cal):
        return self.mse(vis_cal, ir_cal)