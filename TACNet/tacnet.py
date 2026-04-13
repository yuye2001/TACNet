import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from resnet import resnet50, resnet18


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-12)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // reduc_ratio

        self.g = nn.Conv2d(self.in_channels, self.inter_channels, 1, 1, 0)
        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.in_channels, 1, 1, 0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, 1, 1, 0)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, 1, 1, 0)

    def forward(self, x):
        b, c, h, w = x.size()
        g_x = self.g(x).view(b, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(b, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(b, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
        f_div_C = f / (h*w)
        y = torch.matmul(f_div_C, g_x).permute(0, 2, 1).view(b, self.inter_channels, h, w)
        z = self.W(y) + x
        return z


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if m.bias is not None:
            init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()
        model_v = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x

class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()
        model_t = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x

class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()
        model_base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class DCALModule(nn.Module):
    def __init__(self, feat_dim=2048, num_ids=1000, cls_thresh=0.6, consist_thresh=0.45):
        super().__init__()
        self.cls_thresh = cls_thresh
        self.consist_thresh = consist_thresh
        self.classifier = nn.Linear(feat_dim, num_ids)
        self.adaptive_weights = nn.Parameter(torch.ones(num_ids) * 0.5)

    def forward(self, feat, labeled_pids=None):
        cls_logits = self.classifier(feat)
        conf, pseudo_pids = torch.max(F.softmax(cls_logits, dim=-1), dim=-1)
        valid_mask = conf >= self.cls_thresh
        adaptive_w = self.adaptive_weights[pseudo_pids]
        return pseudo_pids, valid_mask, adaptive_w, cls_logits

    def refresh_weights(self):
        self.adaptive_weights.data = torch.clamp(self.adaptive_weights.data, 0.1, 1.0)


class GLPRModule(nn.Module):
    def __init__(self, feat_dim=2048, num_ids=1000, num_local=8):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_local = num_local
        self.momentum = 0.9

        self.global_proto = nn.Parameter(torch.randn(2, num_ids, feat_dim))
        self.local_proto = nn.Parameter(torch.randn(2, num_ids, num_local, feat_dim//num_local))

    def forward(self, feat, modality, pids=None):
        B = feat.shape[0]
        global_feat = feat
        local_feat = feat.view(B, self.num_local, -1)
        refined_feat = feat.clone()
        
        if pids is not None:
            for i in range(B):
                m = modality[i]
                pid = pids[i]
                g_proto = self.global_proto[m, pid]
                l_proto = self.local_proto[m, pid]
                
                self.global_proto.data[m, pid] = self.momentum * g_proto + (1-self.momentum) * global_feat[i]
                self.local_proto.data[m, pid] = self.momentum * l_proto + (1-self.momentum) * local_feat[i]
                
                refined_feat[i] = 0.7 * global_feat[i] + 0.3 * g_proto

        return refined_feat

    def refresh_prototypes(self):
        self.global_proto.data = F.normalize(self.global_proto.data, dim=-1)
        self.local_proto.data = F.normalize(self.local_proto.data, dim=-1)


class CCMModule(nn.Module):
    def __init__(self, feat_dim=2048):
        super().__init__()
        self.proj = nn.Linear(feat_dim, feat_dim)
        self.sim_net = nn.Sequential(nn.Linear(1,16), nn.ReLU(), nn.Linear(16,1), nn.Sigmoid())

    def forward(self, vis_feat, ir_feat):
        vis_p = self.proj(vis_feat)
        ir_p = self.proj(ir_feat)
        
        sim = F.cosine_similarity(vis_p, ir_p, dim=-1, keepdim=True)
        weight = self.sim_net(sim)
        
        vis_cal = weight * ir_p + (1-weight) * vis_p
        ir_cal = weight * vis_p + (1-weight) * ir_p
        return vis_cal, ir_cal, F.mse_loss(vis_cal, ir_cal)

class TACNet(nn.Module):
    def __init__(self, class_num, no_local='on', gm_pool='on', arch='resnet50'):
        super(TACNet, self).__init__()


        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        

        self.non_local = no_local
        if self.non_local == 'on':
            layers=[3,4,6,3]
            non_layers=[0,2,3,0]
            self.NL_1 = nn.ModuleList([Non_local(256) for _ in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0]-(i+1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList([Non_local(512) for _ in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1]-(i+1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList([Non_local(1024) for _ in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2]-(i+1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList([Non_local(2048) for _ in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3]-(i+1) for i in range(non_layers[3])])


        self.pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(self.pool_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.pool_dim, class_num, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.gm_pool = gm_pool


        self.dcal = DCALModule(self.pool_dim, class_num)
        self.glpr = GLPRModule(self.pool_dim, class_num)
        self.ccm = CCMModule(self.pool_dim)


        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def extract_feat(self, x):

        if self.non_local == 'on':
            NL1_counter = 0
            if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
            for i in range(len(self.base_resnet.base.layer1)):
                x = self.base_resnet.base.layer1[i](x)
                if i == self.NL_1_idx[NL1_counter]:
                    x = self.NL_1[NL1_counter](x)
                    NL1_counter += 1

            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet.base.layer2)):
                x = self.base_resnet.base.layer2[i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1

            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.base.layer3)):
                x = self.base_resnet.base.layer3[i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1

            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.base.layer4)):
                x = self.base_resnet.base.layer4[i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1
        else:
            x = self.base_resnet(x)


        if self.gm_pool == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            x_pool = (torch.mean(x**3, dim=-1) + 1e-12)**(1/3)
        else:
            x_pool = self.avgpool(x).flatten(1)

        feat = self.bottleneck(x_pool)
        return x_pool, feat

    def forward(self, x1, x2, modal=0, pids=None):

        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
            modality = torch.cat([torch.zeros(x1.size(0)), torch.ones(x2.size(0))]).long().to(x1.device)
        elif modal == 1:
            x = self.visible_module(x1)
            modality = torch.zeros(x.size(0)).long().to(x.device)
        else:
            x = self.thermal_module(x2)
            modality = torch.ones(x.size(0)).long().to(x.device)


        x_pool, feat = self.extract_feat(x)
        

        refined_feat = self.glpr(feat, modality, pids)
        

        ccm_loss = 0.0
        if modal == 0:
            vis_num = x1.size(0)
            vis_feat = refined_feat[:vis_num]
            ir_feat = refined_feat[vis_num:]
            vis_cal, ir_cal, ccm_loss = self.ccm(vis_feat, ir_feat)
            refined_feat = torch.cat([vis_cal, ir_cal])
        

        if self.training:
            pseudo_pids, valid_mask, adaptive_w, cls_logits = self.dcal(refined_feat)
            return x_pool, refined_feat, self.classifier(refined_feat), ccm_loss, pseudo_pids, valid_mask, adaptive_w
        else:
            return self.l2norm(x_pool), self.l2norm(refined_feat)


embed_net = TACNet