
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from Models1.sync_batchnorm import SynchronizedBatchNorm2d
from torchvision.models import resnet34, resnet50, resnet101, resnet152, resnet18
from Models1.batchnorm import SynchronizedBatchNorm2d
from einops import rearrange, reduce, repeat, parse_shape
from Models1.utils import conv_bn_relu
from torch.nn.modules.utils import _pair
from torch.nn import BatchNorm2d
from torch import Tensor
from pytorch_wavelets import DWTForward
import numpy as np
import networkx as nx

class xin202504a1s1(nn.Module):
    # def __init__(self, backbone,
    #              pretrained=True,
    #              ResNet34M= False,
    #              classes=11):
    def __init__(self, backbone,  sync_bn=True, pretrained=True, ResNet34M= False, criterion=nn.CrossEntropyLoss(ignore_index=255), classes = 24):
        super(xin202504a1s1, self).__init__()
        self.ResNet34M = ResNet34M
        self.backbone = backbone
        self.criterion = criterion

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        if backbone.lower() == "resnet18":
            encoder = resnet18(pretrained=pretrained)
        elif backbone.lower() == "resnet34":
            encoder = resnet34(pretrained=pretrained)
        elif backbone.lower() == "resnet50":
            encoder = resnet50(pretrained=pretrained)
        elif backbone.lower() == "resnet101":
            encoder = resnet101(pretrained=pretrained)
        elif backbone.lower() == "resnet152":
            encoder = resnet152(pretrained=pretrained)
        else:
            raise NotImplementedError("{} Backbone not implemented".format(backbone))

        self.out_channels = [32,64,128,256,512,1024,2048]
        # self.conv1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.conv1_x = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool
        self.conv2_x = encoder.layer1  # 1/4
        self.conv3_x = encoder.layer2  # 1/8
        self.conv4_x = encoder.layer3  # 1/16
        self.conv5_x = encoder.layer4  # 1/32

        # if backbone in ['resnet50','resnet101','resnet152']:
        self.down2 = conv_block(self.out_channels[-4], self.out_channels[1], 3, 1, 1, 1, 1, bn_act=True)
        self.down3 = conv_block(self.out_channels[-3], self.out_channels[2], 3, 1, 1, 1, 1, bn_act=True)
        self.down4 = conv_block(self.out_channels[-2], self.out_channels[3], 3, 1, 1, 1, 1, bn_act=True)
        self.down5 = conv_block(self.out_channels[-1], self.out_channels[4], 3, 1, 1, 1, 1, bn_act=True)

        self.fab = nn.Sequential(
            conv_block(self.out_channels[4],
                       self.out_channels[4]//2,
                       kernel_size = 3,
                       stride= 1,
                       padding=1,
                       group=self.out_channels[4]//2,
                       dilation=1,
                       bn_act=True),
                       nn.Dropout(p=0.3))
        self.ciam =CIAM(self.out_channels[4]//2)  #

        self.cfgb = nn.Sequential(
            conv_block(self.out_channels[4],
                       self.out_channels[4],
                       kernel_size =3,
                       stride= 2,
                       padding = 1,
                       group=self.out_channels[4],
                       dilation=1,
                       bn_act=True),
                       nn.Dropout(p=0.3))

        # self.decoder = DecoderBlock(self.out_channels[4], self.out_channels[4], BatchNorm) #
        self.decoderdowm = conv_block(self.out_channels[4], self.out_channels[4],3,2,padding=1) #
        # self.sca = MultiDirectionalAttention(self.out_channels[4], 4)
        # self.sca = StripAttentionModule(self.out_channels[4], self.out_channels[4])
        # self.sca1 = StripAttentionModule1(self.out_channels[4], self.out_channels[4])
        # self.sca2 = StripAttentionModule2(self.out_channels[4], self.out_channels[4])
        self.d1 = DecoderBlock(self.out_channels[4], self.out_channels[4], BatchNorm)


        self.gfu4 = GlobalFeatureUpsample(self.out_channels[3], self.out_channels[3], self.out_channels[3])
        self.gfu3 = GlobalFeatureUpsample(self.out_channels[2], self.out_channels[3], self.out_channels[2])
        self.gfu2 = GlobalFeatureUpsample(self.out_channels[1], self.out_channels[2], self.out_channels[1])
        self.gfu1 = GlobalFeatureUpsample(self.out_channels[0], self.out_channels[1], self.out_channels[0])


        # self.apf1 = PyrmidFusionNet(self.out_channels[4], self.out_channels[4], self.out_channels[3], classes=classes)
        # self.apf2 = PyrmidFusionNet(self.out_channels[3], self.out_channels[3], self.out_channels[2], classes=classes)
        # self.apf3 = PyrmidFusionNet(self.out_channels[2], self.out_channels[2], self.out_channels[1], classes=classes)
        # self.apf4 = PyrmidFusionNet(self.out_channels[1], self.out_channels[1], self.out_channels[1], classes=classes)  ##xliang
        
        self.apf1 = PyrmidFusionNet(self.out_channels[4], self.out_channels[4], self.out_channels[3], classes=classes)
        self.apf2 = PyrmidFusionNet(self.out_channels[3], self.out_channels[3], self.out_channels[2], classes=classes)
        self.apf3 = PyrmidFusionNet(self.out_channels[2], self.out_channels[2], self.out_channels[1], classes=classes)
        self.apf4 = PyrmidFusionNet(self.out_channels[1], self.out_channels[1], self.out_channels[1], classes=classes)



        self.classifier1 = SegHead(self.out_channels[3], classes)
        self.classifier0 = SegHead(self.out_channels[0], classes)
        self.classifier2 = SegHead(self.out_channels[2], classes)
        self.classifier3 = SegHead(self.out_channels[1], classes)
        self.classifier4 = SegHead(self.out_channels[4], classes)

        self.classifier = SegHead(self.out_channels[1], classes)

        self.D = LCGB(3, self.out_channels[1])
        # self.saff = SAFF()

        # self.Dlv = TopoGeomConv(self.out_channels[4], self.out_channels[4], 3)
        # self.ESRA1 = ESRA(self.out_channels[1])
        # self.LECA1 = LECA(self.out_channels[1])
        # self.LECA2 = LECA(self.out_channels[2])
        # self.LECA3 = LECA(self.out_channels[3])
        # self.LECA4 = LECA(self.out_channels[4])
        self.LECA1 = LECA1(self.out_channels[0])
        self.LECA2 = LECA1(self.out_channels[4])
        self.LECAlv = LECAlv(self.out_channels[0])
        self.QKVAttention = QKVAttention(self.out_channels[4],sparsity=0.2)
        # self.Partial_conv3 = Partial_conv3(self.out_channels[4], 8, 'split_cat')

        self.Partial_conv3 = Partial_conv3(self.out_channels[4], 8, 'split_cat')
        self.sf = StripeFeatureExtractor(self.out_channels[4], self.out_channels[4])
        self.QKVlvv = QKVAttentionlvv(self.out_channels[4],sparsity=0.2,std=3.0)
        self.block = block(self.out_channels[1])
        # self.GraphSparsifiedSelfAttention = GraphSparsifiedSelfAttention(self.out_channels[4], keep_topk=8)
        self.GaussianFourierAttention = GaussianFourierAttention(self.out_channels[4], sigma=1.0)

    def forward(self, x, y=None):
        B, C, H, W = x.size()

        # DH = self.D(x) #torch.Size([6, 64, 160, 160])
        # print(DH.size())

        x = self.conv1_x(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.maxpool(x1)
        if self.ResNet34M:
            x2 = self.conv2_x(x1)
        else:
            x2 = self.conv2_x(x)
        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)

        if self.backbone in ['resnet50', 'resnet101', 'resnet152']:
            x2 = self.down2(x2)
            x3 = self.down3(x3)
            x4 = self.down4(x4)
            x5 = self.down5(x5)
# ########
#         x21 = self.ESRA1(x2)
#         x22 = self.LECA1(x2)
#         # x231 = torch.add(x21, x22)
#         x232 = torch.add(x231, x2)
#         print(x21.size())  #torch.Size([6, 64, 160, 160])
#         print(x22.size()) #torch.Size([6, 64, 160, 160])
# ########
        # print(x5.size())  #torch.Size([6, 512, 16, 16])
        # CFGB = self.cfgb(x5) #torch.Size([6, 512, 8, 8])
        # print(CFGB.size())
        # cfgb10 = self.sca1(x5) #torch.Size([6, 512, 16, 16])
        # cfgb11 = self.sca2(x5)  # torch.Size([6, 512, 16, 16])
        # cfgb1 = cfgb10 + cfgb11 - x5
        # cfgb1 = self.Dlv(x5)
        # print(cfgb1.size())
        # s1 = self.sca(x5)
        # CFGB =self.decoderdowm(x5) #torch.Size([6, 512, 8, 8])
        # print(cfgb1d.size())
        # print(x2.size()) #torch.Size([6, 64, 160, 160])
        # print(x3.size()) #torch.Size([6, 128, 80, 80])
        # print(x4.size()) #orch.Size([6, 256, 40, 40])
        # print(x5.size()) #torch.Size([6, 512, 20, 20])


        # D112 = torch.add(DH, x2) #torch.Size([6, 64, 160, 160])
        # print(D112.size())
        # L1 = self.LECA4(x5) + x5
        # L2 = self.LECA3(x4) + x4
        # L3 = self.LECA3(x3) + x3
        # L4 = self.LECA3(x2) + x2

        # CFGB1 = self.d1(x5)
        # CFGB = self.LECA4(x5) + x5
        # CFGB = self.QKVAttention(x5)
        # CFGB = self.Partial_conv3(x5)
        # CFGB = self.LECA2(x5) + x5
        # CFGB = self.Partial_conv3(x5)
        # CFGB = self.sf(x5)
        # CFGB = self.QKVlvv(x5)
        # CFGB = self.GraphSparsifiedSelfAttention(x5)
        CFGB = self.GaussianFourierAttention(x5)



        APF1, cls1 = self.apf1(CFGB, x5)

        APF2, cls2 = self.apf2(APF1, x4)

        APF3, cls3 = self.apf3(APF2, x3)

        APF4, cls4 = self.apf4(APF3, x2) #x2

        # SAW, CAW = self.saff(APF4)
        # SA = SAW.unsqueeze(2).unsqueeze(3) * APF4
        # CA = CAW.unsqueeze(2).unsqueeze(3) * APF4
        # APF4 = SA + CA + APF4

        # AW = self.saff(APF4)
        # AW = AW.unsqueeze(2).unsqueeze(3)
        # AW = AW.expand_as(APF4)
        #
        # APF4 = APF4 +  AW * APF4
        # APF4 = torch.cat([APF4, AW*APF4],1)
        # print("WWWWWW")
        # FAB = self.fab(x5) #torch.Size([6, 256, 16, 16])
        # print(FAB.size()) #torch.Size([6, 256, 16, 16])
        # fab = self.ciam(x5)
        # print(fab.size())

        # dec5 = self.gfu4(APF1, FAB)
        # dec4 = self.gfu3(APF2, dec5)
        # dec3 = self.gfu2(APF3, dec4)
        # dec2 = self.gfu1(APF4, dec3)
        # A5 = self.LECA1(APF4)
        # A5 = self.LECAlv(APF4) + x2
        # A51 = torch.cat([A5, x2],1)
        # APF41 = self.block(APF4)

        classifier = self.classifier(APF4)

        # sup1 = F.interpolate(cls1, size=(H, W), mode="bilinear", align_corners=True)
        # sup2 = F.interpolate(cls2, size=(H, W), mode="bilinear", align_corners=True)
        # sup3 = F.interpolate(cls3, size=(H, W), mode="bilinear", align_corners=True)
        # sup4 = F.interpolate(cls4, size=(H, W), mode="bilinear", align_corners=True)
        predict = F.interpolate(classifier, size=(H, W), mode="bilinear", align_corners=True)

        if self.training:
            main_loss = self.criterion(predict, y)
            return predict.max(1)[1], main_loss, main_loss
        else:
            return predict

        return predict
        # if self.training:
        #     return predict, sup1, sup2, sup3, sup4
        # else:
        #     return predict
class GaussianFourierAttention(nn.Module):
    def __init__(self, dim, sigma=1.0):
        """
        dim: 通道数 C
        sigma: 高斯傅里叶权重标准差
        """
        super(GaussianFourierAttention, self).__init__()
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.out_proj = nn.Conv2d(dim, dim, 1)
        self.sigma = sigma

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W

        # Q, K, V
        Q = self.q_proj(x).flatten(2).transpose(1, 2)  # (B, N, C)
        K = self.k_proj(x).flatten(2).transpose(1, 2)  # (B, N, C)
        V = self.v_proj(x).flatten(2).transpose(1, 2)  # (B, N, C)

        # 原始注意力
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (C ** 0.5)  # (B, N, N)

        # ========== 高斯傅里叶生成 mask ==========
        # 构建坐标矩阵 (H*W, 2)
        coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=-1).reshape(-1,2).float().to(x.device)  # (N,2)
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N,N,2)
        dist2 = (diff ** 2).sum(-1)  # (N,N) 距离平方

        # 高斯傅里叶 mask
        mask = torch.exp(-dist2 / (2 * self.sigma**2))  # (N,N)
        mask = mask.unsqueeze(0).repeat(B,1,1)  # (B,N,N)

        # 应用 mask 优化注意力
        attn = attn * mask
        attn = F.softmax(attn, dim=-1)

        # 输出
        out = torch.matmul(attn, V)  # (B, N, C)
        out = out.transpose(1,2).view(B, C, H, W)
        out = self.out_proj(out)

        return out



class GraphSparsifiedSelfAttention(nn.Module):
    def __init__(self, dim, keep_topk=5):
        """
        dim: 通道数 C
        keep_topk: 每个节点保留的最大边数
        """
        super(GraphSparsifiedSelfAttention, self).__init__()
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.keep_topk = keep_topk

    def forward(self, x):
        """
        x: (B, C, H, W)
        return:
            out: (B, C, H, W)
            attn_full: (B, N, N) 原始注意力
            attn_sparse: (B, N, N) 稀疏化后的注意力
        """
        B, C, H, W = x.size()
        N = H * W

        # 计算 Q, K, V
        Q = self.q_proj(x).flatten(2).transpose(1, 2)  # (B, N, C)
        K = self.k_proj(x).flatten(2).transpose(1, 2)  # (B, N, C)
        V = self.v_proj(x).flatten(2).transpose(1, 2)  # (B, N, C)

        # 原始注意力矩阵
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (C ** 0.5)  # (B, N, N)
        attn = F.softmax(attn, dim=-1)

        # ========= 基于图的稀疏化 =========
        attn_sparse = []
        for b in range(B):
            A = attn[b].detach().cpu().numpy()  # (N, N)
            G = nx.from_numpy_array(A)

            # (1) 最小生成树保证连通性
            mst = nx.minimum_spanning_tree(G)
            mst_adj = nx.to_numpy_array(mst)

            # (2) 基于度的修剪：每个节点保留 top-k 边
            for i in range(N):
                row = A[i]
                top_idx = row.argsort()[-self.keep_topk:]  # 选出 top-k 边
                mask = np.zeros_like(row)
                mask[top_idx] = 1
                mst_adj[i] = np.maximum(mst_adj[i], mask)

            attn_sparse.append(torch.tensor(mst_adj, dtype=torch.float32, device=x.device))

        attn_sparse = torch.stack(attn_sparse, dim=0)  # (B, N, N)

        # ========= 稀疏注意力应用 =========
        out = torch.matmul(attn_sparse, V)  # (B, N, C)
        out = out.transpose(1, 2).view(B, C, H, W)  # 还原回 (B, C, H, W)
        out = self.out_proj(out)

        return out

        # return out, attn, attn_sparse


class block(nn.Module):
    def __init__(self, dim, r=16, L=32):
        super().__init__()
        d = max(dim // r, L)
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, dim, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        dim = x.size(1)
        attn1 = self.conv0(x)  # conv_3*3
        attn2 = self.conv_spatial(attn1)  # conv_3*3 -> conv_5*5

        attn1 = self.conv1(attn1) # b, dim/2, h, w
        attn2 = self.conv2(attn2) # b, dim/2, h, w

        attn = torch.cat([attn1, attn2], dim=1)  # b,c,h,w
        avg_attn = torch.mean(attn, dim=1, keepdim=True) # b,1,h,w
        max_attn, _ = torch.max(attn, dim=1, keepdim=True) # b,1,h,w
        agg = torch.cat([avg_attn, max_attn], dim=1) # spa b,2,h,w

        ch_attn1 = self.global_pool(attn) # b,dim,1, 1
        z = self.fc1(ch_attn1)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, 2, dim // 2, -1)
        a_b = self.softmax(a_b)

        a1,a2 =  a_b.chunk(2, dim=1)
        a1 = a1.reshape(batch_size,dim // 2,1,1)
        a2 = a2.reshape(batch_size, dim // 2, 1, 1)

        w1 = a1 * agg[:, 0, :, :].unsqueeze(1)
        w2 = a2 * agg[:, 0, :, :].unsqueeze(1)

        attn = attn1 * w1 + attn2 * w2
        attn = self.conv(attn).sigmoid()

        return (x * attn + x)


class QKVAttentionlvv(nn.Module):
    def __init__(self, in_channels, sparsity=0.1, std=3.0):
        super(QKVAttentionlvv, self).__init__()
        self.q_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sparsity = sparsity  # Fraction of values to retain in the sparse attention
        self.sflv = StripeFeatureExtractor(in_channels,in_channels)
        self.std = std  # Gaussian 控制范围

    def build_gaussian_bias(self, H, W, device):
        """构建 [H*W, H*W] 的高斯空间偏置矩阵"""
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        coords = torch.stack([y, x], dim=-1).float().to(device)  # [H, W, 2]
        coords_flat = coords.view(-1, 2)  # [H*W, 2]
        dist = torch.cdist(coords_flat, coords_flat, p=2)  # [H*W, H*W]
        bias = torch.exp(- (dist ** 2) / (2 * self.std ** 2))  # 高斯偏置
        return bias  # [H*W, H*W]

    def compute_entropy(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1)  # Flatten to [B, C, H*W]
        norm_feature = torch.softmax(x, dim=2)
        entropy = -torch.sum(norm_feature * torch.log(norm_feature + 1e-8), dim=2)
        return entropy.view(B, 1, H * W)  # [B, 1, H*W]

    def forward(self, x):
        B, C, H, W = x.size()

        # 条纹引导特征
        x1 = self.sflv(x)  # 用于生成 Key

        # Q, K, V
        Q = self.q_conv(x).view(B, H * W, C)  # [B, HW, C]
        K = self.k_conv(x1).view(B, C, H * W)  # [B, C, HW]
        V = self.v_conv(x).view(B, C, H * W)  # [B, C, HW]

        # 原始注意力
        affinity_matrix = torch.bmm(Q, K) / (C ** 0.5)  # [B, HW, HW]

        # Apply sparsity mask
        topk = int(self.sparsity * affinity_matrix.size(1))  # Number of top elements to keep
        values, indices = torch.topk(affinity_matrix, topk, dim=-1, largest=True, sorted=False)
        mask = torch.zeros_like(affinity_matrix)
        mask.scatter_(2, indices, 1.0)  # Scatter the top k values
        sparse_affinity_matrix = affinity_matrix * mask  # Apply the mask

        optimized_v = V.view(B, C, H * W)  # [B, C, H*W]

        # Weighted sum
        weighted_sum = torch.matmul(sparse_affinity_matrix, optimized_v.permute(0, 2, 1))  # [B, H*W, C]

        # Reshape and add input
        output = weighted_sum.permute(0, 2, 1).view(B, C, H, W) + x  # [B, C, H, W]

        return output

        # # 添加 Gaussian Bias
        # gaussian_bias = self.build_gaussian_bias(H, W, x.device)  # [HW, HW]
        # affinity_matrix = affinity_matrix + gaussian_bias.unsqueeze(0)  # [B, HW, HW]
        #
        # # Softmax 注意力分布
        # attn = F.softmax(affinity_matrix, dim=-1)  # [B, HW, HW]
        #
        # # 加权 V
        # weighted_sum = torch.bmm(attn, V.permute(0, 2, 1))  # [B, HW, C]
        # output = weighted_sum.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]
        #
        # return output + x  # 残差连接
    #         B, C, H, W = x.size()
    #
    #         # Compute Q, K, V
    #         Q = self.q_conv(x).view(B, H * W, C)  # [B, H*W, C]
    #         K = self.k_conv(x).view(B, C, H * W)  # [B, C, H*W]
    #         V = self.v_conv(x)  # [B, C, H, W]
    #
    #         # Compute affinity matrix
    #         affinity_matrix = torch.matmul(Q, K)  # [B, H*W, H*W]
    #
    #         # Apply sparsity mask
    #         topk = int(self.sparsity * affinity_matrix.size(1))  # Number of top elements to keep
    #         values, indices = torch.topk(affinity_matrix, topk, dim=-1, largest=True, sorted=False)
    #         mask = torch.zeros_like(affinity_matrix)
    #         mask.scatter_(2, indices, 1.0)  # Scatter the top k values
    #         sparse_affinity_matrix = affinity_matrix * mask  # Apply the mask
    #
    #         optimized_v = V.view(B, C, H * W)  # [B, C, H*W]
    #
    #         # Weighted sum
    #         weighted_sum = torch.matmul(sparse_affinity_matrix, optimized_v.permute(0, 2, 1))  # [B, H*W, C]
    #
    #         # Reshape and add input
    #         output = weighted_sum.permute(0, 2, 1).view(B, C, H, W) + x  # [B, C, H, W]
    #
    #         return output
    # def forward(self, x):
    #     B, C, H, W = x.size()
    #     # print(x.size())
    #
    #     x1 = self.sflv(x)
    #     # print(x1.size())
    #     # Compute Q, K, V
    #     Q = self.q_conv(x).view(B, H * W, C)  # [B, H*W, C]
    #     K = self.k_conv(x1).view(B, C, H * W)  # [B, C, H*W]
    #     V = self.v_conv(x)  # [B, C, H, W]
    #
    #     # Compute affinity matrix
    #     affinity_matrix = torch.matmul(Q, K)  # [B, H*W, H*W]
    #
    #     optimized_v = V.view(B, C, H * W)
    #
    #     weighted_sum = torch.matmul(affinity_matrix, optimized_v.permute(0, 2, 1))  # [B, H*W, C]
    #     # Reshape and add input
    #     output = weighted_sum.permute(0, 2, 1).view(B, C, H, W) + x  # [B, C, H, W]
    #
    #     return output

class StripeFeatureExtractor(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(StripeFeatureExtractor, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        self.conv_HL = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_LH = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_HH = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # 可以最后加一个融合卷积
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        yL, yH = self.wt(x)  # 小波分解
        y_HL = yH[0][:, :, 0, :, :]  # 水平变化（竖条）
        y_LH = yH[0][:, :, 1, :, :]  # 垂直变化（横条）
        y_HH = yH[0][:, :, 2, :, :]  # 对角变化（斜条）

        # 上采样回原尺寸
        yL = F.interpolate(yL, size=(H, W), mode='bilinear', align_corners=False)
        y_HL = F.interpolate(y_HL, size=(H, W), mode='bilinear', align_corners=False)
        y_LH = F.interpolate(y_LH, size=(H, W), mode='bilinear', align_corners=False)
        y_HH = F.interpolate(y_HH, size=(H, W), mode='bilinear', align_corners=False)

        # feat_HL = self.conv_HL(y_HL)
        # feat_LH = self.conv_LH(y_LH)
        # feat_HH = self.conv_HH(y_HH)
        x = torch.cat([y_HL, y_LH, y_HH], dim=1)

        # # 把不同方向的条纹特征拼接起来
        # x = torch.cat([feat_HL, feat_LH, feat_HH], dim=1)
        x = self.fuse(x)

        return x

class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.QKVAttention = QKVAttention(self.dim_conv3, sparsity=0.2)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        ####yuanlai####
        # x1 = self.partial_conv3(x1)
        ###############
        x1 = self.QKVAttention(x1)

        x = torch.cat((x1, x2), 1)

        return x

class QKVAttention(nn.Module):
    def __init__(self, in_channels, sparsity=0.1):
        super(QKVAttention, self).__init__()
        self.q_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sparsity = sparsity  # Fraction of values to retain in the sparse attention
        self.LECALW = LECALW(in_channels)

    def forward(self, x):
        B, C, H, W = x.size()

        # Compute Q, K, V
        Q = self.q_conv(x).view(B, H * W, C)  # [B, H*W, C]
        K = self.k_conv(x).view(B, C, H * W)  # [B, C, H*W]
        V = self.v_conv(x)  # [B, C, H, W]

        # Compute affinity matrix
        affinity_matrix = torch.matmul(Q, K)  # [B, H*W, H*W]

        # Compute entropy
        entropy = self.LECALW(x)  # [B, C, 1, 1]
        entropy = entropy.view(B, C)  # [B, C]

        # Calculate the number of elements to retain
        max_k = min(int(self.sparsity * H * W), H * W)  # Ensure max_k <= H * W
        max_k = min(max_k, C)  # Ensure max_k <= C
        max_k = max(max_k, 1)  # Ensure max_k >= 1

        # Get the threshold for the top-k values
        threshold = torch.topk(entropy, max_k, dim=-1).values.min()  # [B]

        # Create a mask for the affinity matrix
        mask = (entropy >= threshold.unsqueeze(-1)).float()  # [B, C]

        # Expand the mask to [B, C, H*W]
        mask = mask.unsqueeze(-1).expand(B, C, H * W)  # [B, C, H*W]

        # Reshape the mask to [B, C, H, W]
        mask = mask.view(B, C, H, W)  # [B, C, H, W]

        # Apply the mask to the affinity matrix
        # Expand mask to [B, H*W, H*W] by broadcasting
        mask = mask.view(B, C, H * W)  # [B, C, H*W]
        mask = mask.permute(0, 2, 1)  # [B, H*W, C]
        mask = mask.unsqueeze(-1).expand(B, H * W, C, H * W)  # [B, H*W, C, H*W]
        mask = mask.permute(0, 1, 3, 2)  # [B, H*W, H*W, C]
        mask = mask.mean(dim=-1)  # [B, H*W, H*W]

        sparse_affinity_matrix = affinity_matrix * mask  # [B, H*W, H*W]

        # Weighted sum
        optimized_v = V.view(B, C, H * W)  # [B, C, H*W]
        weighted_sum = torch.matmul(sparse_affinity_matrix, optimized_v.permute(0, 2, 1))  # [B, H*W, C]

        # Reshape and add input
        output = weighted_sum.permute(0, 2, 1).view(B, C, H, W) + x  # [B, C, H, W]

        return output

class LECALW(nn.Module):
    def __init__(self, channel, ratio=16):
        super(LECALW, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(channel)

    def compute_entropy(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1)  # [B, C, H*W]
        norm_feature = torch.softmax(x, dim=2)
        entropy = -torch.sum(norm_feature * torch.log(norm_feature + 1e-8), dim=2)  # [B, C]
        return entropy.view(B, C, 1, 1)  # [B, C, 1, 1]

    def forward(self, x):
        B, C, H, W = x.size()
        max_value, _ = torch.max(x.view(B, C, -1), dim=2, keepdim=True)
        min_value, _ = torch.min(x.view(B, C, -1), dim=2, keepdim=True)
        diff = max_value - min_value
        norm_feature = x / (diff.view(B, C, 1, 1) + 1e-8)
        x2 = x * norm_feature

        entropy = self.compute_entropy(x2)
        tb = self.norm(entropy)
        entropy_rate_feature = self.sigmoid(tb)

        return (entropy_rate_feature)


class LECA1(nn.Module):
    def __init__(self, channel, ratio=16):
        super(LECA1, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(channel)

    def compute_entropy(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1)  # [B, C, H*W]
        norm_feature = torch.softmax(x, dim=2)
        entropy = -torch.sum(norm_feature * torch.log(norm_feature + 1e-8), dim=2)  # [B, C]
        return entropy.view(B, C, 1, 1)  # [B, C, 1, 1]

    def forward(self, x):
        B, C, H, W = x.size()
        max_value, _ = torch.max(x.view(B, C, -1), dim=2, keepdim=True)
        min_value, _ = torch.min(x.view(B, C, -1), dim=2, keepdim=True)
        diff = max_value - min_value
        norm_feature = x / (diff.view(B, C, 1, 1) + 1e-8)
        x2 = x * norm_feature

        entropy = self.compute_entropy(x2)
        tb = self.norm(entropy)
        entropy_rate_feature = self.sigmoid(tb)

        return (entropy_rate_feature * x)

class LECAlv(nn.Module): #空间信息熵注意力特征
    def __init__(self, channel, ratio=16):
        super(LECAlv, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(1)

    def compute_entropylv(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1)  # [B, C, H*W]
        norm_feature = torch.softmax(x, dim=1)  # 在通道维度上归一化
        entropy = -torch.sum(norm_feature * torch.log(norm_feature + 1e-8), dim=1)  # [B, H*W]
        return entropy.view(B, 1, H, W)  # [B, 1, H, W]

    def forward(self, x):
        B, C, H, W = x.size()
        max_value, _ = torch.max(x.view(B, C, -1), dim=2, keepdim=True)
        min_value, _ = torch.min(x.view(B, C, -1), dim=2, keepdim=True)
        diff = max_value - min_value
        norm_feature = x / (diff.view(B, C, 1, 1) + 1e-8)
        x2 = x * norm_feature

        entropy = self.compute_entropylv(x2)
        tb = self.norm(entropy)
        entropy_rate_feature = self.sigmoid(tb)

        return (entropy_rate_feature * x)


# class QKVAttention(nn.Module):
#     def __init__(self, in_channels, sparsity=0.1):
#         super(QKVAttention, self).__init__()
#         self.q_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.k_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.sparsity = sparsity  # Fraction of values to retain in the sparse attention
#
#     def compute_entropy(self, x):
#         B, C, H, W = x.size()
#         x = x.view(B, C, -1)  # Flatten to [B, C, H*W]
#         norm_feature = torch.softmax(x, dim=2)
#         entropy = -torch.sum(norm_feature * torch.log(norm_feature + 1e-8), dim=2)
#         return entropy.view(B, 1, H * W)  # [B, 1, H*W]
#
#     def forward(self, x):
#         B, C, H, W = x.size()
#
#         # Compute Q, K, V
#         Q = self.q_conv(x).view(B, H * W, C)  # [B, H*W, C]
#         K = self.k_conv(x).view(B, C, H * W)  # [B, C, H*W]
#         V = self.v_conv(x)  # [B, C, H, W]
#
#         # Compute affinity matrix
#         affinity_matrix = torch.matmul(Q, K)  # [B, H*W, H*W]
#
#         # Apply sparsity mask
#         topk = int(self.sparsity * affinity_matrix.size(1))  # Number of top elements to keep
#         values, indices = torch.topk(affinity_matrix, topk, dim=-1, largest=True, sorted=False)
#         mask = torch.zeros_like(affinity_matrix)
#         mask.scatter_(2, indices, 1.0)  # Scatter the top k values
#         sparse_affinity_matrix = affinity_matrix * mask  # Apply the mask
#
#         optimized_v = V.view(B, C, H * W)  # [B, C, H*W]
#
#         # Weighted sum
#         weighted_sum = torch.matmul(sparse_affinity_matrix, optimized_v.permute(0, 2, 1))  # [B, H*W, C]
#
#         # Reshape and add input
#         output = weighted_sum.permute(0, 2, 1).view(B, C, H, W) + x  # [B, C, H, W]
#
#         return output


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp

        self.deconv1 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )

        self.bn2 = BatchNorm(in_channels // 4 + in_channels // 4)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x, inp = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        if self.inp:
            x = F.interpolate(x, scale_factor=2)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)

class DirectionalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, direction):
        super(DirectionalConv2d, self).__init__()
        self.direction = direction
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Set weights based on the direction
        with torch.no_grad():
            if direction == 'horizontal':
                self.weight[:, :, kernel_size // 2, :] = 1.0
            elif direction == 'vertical':
                self.weight[:, :, :, kernel_size // 2] = 1.0
            elif direction == 'main_diagonal':
                for i in range(kernel_size):
                    self.weight[:, :, i, i] = 1.0
            elif direction == 'anti_diagonal':
                for i in range(kernel_size):
                    self.weight[:, :, i, kernel_size - i - 1] = 1.0

    def forward(self, x):
        x = F.conv2d(x, self.weight, self.bias, stride=1, padding=self.kernel_size // 2)
        x = torch.nan_to_num(x)  # Replace NaN with zero
        return x


class MultiDirectionalAttention(nn.Module):
    def __init__(self, in_channels, num_directions=4):
        super(MultiDirectionalAttention, self).__init__()
        self.in_channels = in_channels
        self.num_directions = num_directions
        self.directional_convs_q = nn.ModuleList()
        self.directional_convs_k = nn.ModuleList()

        directions = ['horizontal', 'vertical', 'main_diagonal', 'anti_diagonal']

        for i in range(num_directions):
            self.directional_convs_q.append(
                DirectionalConv2d(in_channels, in_channels // num_directions, 3, directions[i % len(directions)]))
            self.directional_convs_k.append(
                DirectionalConv2d(in_channels, in_channels // num_directions, 3, directions[i % len(directions)]))

    def forward(self, x):
        # Extract features for Q using multiple directional convolutions
        q_features = [conv(x) for conv in self.directional_convs_q]
        q = torch.cat(q_features, dim=1)  # Concatenate along the channel dimension

        # Extract features for K using multiple directional convolutions
        k_features = [conv(x) for conv in self.directional_convs_k]
        k = torch.cat(k_features, dim=1)  # Concatenate along the channel dimension

        # Reshape Q and K to (B, C, H*W)
        B, C, H, W = q.size()
        q = q.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        k = k.view(B, C, -1)  # (B, C, H*W)

        # Compute Attention Map
        attn = torch.bmm(q, k)
        attn = F.softmax(attn, dim=-1)

        # V remains the same as input
        v = x.view(B, C, -1)  # (B, C, H*W)

        # Apply Attention to V
        direction_constraint = torch.bmm(attn, v.transpose(1, 2))

        # Replace NaN and Inf with finite values
        direction_constraint = torch.nan_to_num(direction_constraint)

        # Reshape back to (B, C, H, W)
        direction_constraint = direction_constraint.transpose(1, 2).view(B, C, H, W)

        # Combine with input feature
        out = direction_constraint + x
        out = torch.nan_to_num(out)  # Replace NaN with zero

        return out


class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, deformable_groups=1):
        super(DeformConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size * deformable_groups, kernel_size=3, padding=1)

    def forward(self, x):
        offset = self.offset_conv(x)
        return self.conv(x, offset)

class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SelfAttention, self).__init__()
        self.q_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, q, k, v):
        B, C, H, W = q.size()
        proj_q = self.q_conv(q).view(B, -1, H * W).permute(0, 2, 1)  # B, HW, C
        proj_k = self.k_conv(k).view(B, -1, H * W)  # B, C, HW
        proj_v = self.v_conv(v).view(B, -1, H * W).permute(0, 2, 1)  # B, HW, C

        energy = torch.bmm(proj_q, proj_k)  # B, HW, HW
        attention = F.softmax(energy, dim=-1)  # B, HW, HW

        out = torch.bmm(attention, proj_v)  # B, HW, C
        out = out.view(B, C, H, W)

        return self.gamma * out + v


def compute_entropy(x):
    # Flatten the 2D tensor
    batch_size, num_channels, height, width = x.size()
    flat_feature_map = x.view(batch_size, num_channels, -1)

    # Calculate channel entropies
    channel_entropies = -torch.sum(flat_feature_map * torch.log2(flat_feature_map + 1e-10), dim=2)

    return channel_entropies


class LECA(nn.Module):
    def __init__(self, channel, ratio=16):
        super(LECA, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm2d(channel)

    def forward(self, x):
        ###################### entropy compute ###################
        x1 = x.clone()
        chan = int(x1.shape[1])
        box = int(x1.shape[0])

        # Normalize the feature map
        max_value, _ = torch.max(x1.view(box, chan, -1), dim=2, keepdim=True)
        min_value, _ = torch.min(x1.view(box, chan, -1), dim=2, keepdim=True)
        diff = max_value - min_value
        norm_feature = x1 / (diff.view(box, chan, 1, 1) + 1e-10)

        # Calculate entropy
        x2 = x1 * norm_feature
        entropy = compute_entropy(x2)
        entropy = entropy.view(box, chan, 1, 1)

        # Apply normalization and sigmoid activation
        tb = self.norm(entropy)
        entropy_rate_feature1 = self.sigmoid(tb)

        # Multiply the original input by the entropy_rate_feature to match dimensions
        output = x * entropy_rate_feature1

        ###################### reliability compute ###################
        # Calculate evidence and reliability scores
        evidence = torch.sum(entropy_rate_feature1, dim=(1, 2, 3), keepdim=True)
        reliability = 1 - torch.tanh(evidence / (torch.sum(evidence, dim=0, keepdim=True) + 1e-10))

        # Normalize reliability score
        reliability_normalized = reliability / (torch.sum(reliability, dim=0, keepdim=True) + 1e-10)

        # Adjust output using reliability scores
        output_adjusted = output * reliability_normalized

        # Add adjusted output to the original output
        final_output = output + output_adjusted

        return final_output


class EntorylAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(EntorylAttentionModule, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm2d(channel)

    def forward(self, x):
        ######################entorypycompute###################
        x1 = x.clone()
        chan = int(x1.shape[1])
        box = int(x1.shape[0])

        max_value, _ = torch.max(x1.view(box, chan, -1), dim=2, keepdim=True)
        min_value, _ = torch.min(x1.view(box, chan, -1), dim=2, keepdim=True)
        diff = max_value - min_value
        norm_feature = x1 / diff.view(box, chan, 1, 1)
        x2 = x1 * norm_feature

        entropy = compute_entropy(x2)
        entropy = entropy.view(box, chan, 1, 1)

        tb = self.norm(entropy)
        entrop_yrate_feature1 = self.sigmoid(tb)

        return entrop_yrate_feature1

class ESRA(nn.Module):
    def __init__(self, channel, droup=0.0):
        super(ESRA, self).__init__()

        self.entory_attention = EntorylAttentionModule(channel)

        self.conv2d = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(normalized_shape = [1,1])
        self.dropout = nn.Dropout(droup)
        # self.softmax = nn.Softmax()
        self.softmax = nn.Softmax()

        self.sigmoid = nn.Sigmoid()
        self.num_heads = 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
    def forward(self, x):

        entory_value=self.entory_attention(x)*x
        x1=torch.mean(x, dim=1, keepdim=True)
        x2, _=torch.max(x, dim=1, keepdim=True)


        Q=x1.permute(0, 2, 1, 3)
        K=self.conv2d(entory_value).permute(0, 2, 1, 3)
        T=x2.permute(0, 2, 1, 3)
        V=x.permute(0, 2, 1, 3)


        attn1 = Q @ K.transpose(2, 3) * (x.shape[-1] ** -0.5)
        attn1 = attn1.softmax(dim=-1)

        attn2 = Q @ T.transpose(2, 3) * (x.shape[-1] ** -0.5)
        attn2 = attn2.softmax(dim=-1)

        attn3 = T @ K.transpose(2, 3) * (x.shape[-1] ** -0.5)
        attn3 = attn3.softmax(dim=-1)

        attn=attn1+attn2+attn3

        # 计算加权后的注意力矩阵的证据不确定性评估的可靠性分数
        evidence = torch.sum(attn, dim=(1, 2), keepdim=True)
        reliability = 1 - torch.tanh(evidence / torch.sum(evidence, dim=(1, 2), keepdim=True))

        # 归一化可靠性分数
        reliability_normalized = reliability / torch.sum(reliability, dim=(1, 2), keepdim=True)

        # Apply normalized reliability score to attention
        attn11 = attn * reliability_normalized
        attn21 = attn11.softmax(dim=1)

        # # 对加权后的注意力权重进行softmax归一化
        # attn = self.softmax(attn)

        out = (attn21 * V).permute(0, 2, 1, 3)
        return out

class QKV(nn.Module):
    def __init__(self, in_channels, sparsity=0.1):
        super(QKV, self).__init__()
        self.q_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.leca = LECA1(in_channels)
        self.sparsity = sparsity  # Fraction of values to retain in the sparse attention

    def compute_entropy(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1)  # Flatten to [B, C, H*W]
        norm_feature = torch.softmax(x, dim=2)
        entropy = -torch.sum(norm_feature * torch.log(norm_feature + 1e-8), dim=2)
        return entropy.view(B, 1, H * W)  # [B, 1, H*W]

    def forward(self, x, y):
        B, C, H, W = x.size()

        # Compute Q, K, V
        Q = self.q_conv(y).view(B, H * W, C)  # [B, H*W, C]
        K = self.k_conv(x).view(B, C, H * W)  # [B, C, H*W]
        V = self.v_conv(x)  # [B, C, H, W]

        # Compute affinity matrix
        # affinity_matrix = torch.matmul(Q, K)  # [B, H*W, H*W]
        sparse_affinity_matrix = torch.matmul(Q, K)  # [B, H*W, H*W]

        # # Apply sparsity mask
        # topk = int(self.sparsity * affinity_matrix.size(1))  # Number of top elements to keep
        # values, indices = torch.topk(affinity_matrix, topk, dim=-1, largest=True, sorted=False)
        # mask = torch.zeros_like(affinity_matrix)
        # mask.scatter_(2, indices, 1.0)  # Scatter the top k values
        # sparse_affinity_matrix = affinity_matrix * mask  # Apply the mask

        # # LECA1 optimization on V channel
        # optimized_v = self.leca(V).view(B, C, H * W)  # [B, C, H*W]
        optimized_v = V.view(B, C, H * W)  # [B, C, H*W]

        # Weighted sum
        weighted_sum = torch.matmul(sparse_affinity_matrix, optimized_v.permute(0, 2, 1))  # [B, H*W, C]

        # Reshape and add input
        output = weighted_sum.permute(0, 2, 1).view(B, C, H, W) + x  # [B, C, H, W]

        return output



class PyrmidFusionNet(nn.Module):
    def __init__(self, channels_high, channels_low, channel_out, classes=11):
        super(PyrmidFusionNet, self).__init__()

        self.lateral_low = conv_block(channels_low, channels_high, 1, 1, bn_act=True, padding=0)

        self.conv_low = conv_block(channels_high, channel_out, 3, 1, bn_act=True, padding=1)
        self.sa = SpatialAttention(channel_out, channel_out)
        self.salw = SpatialAttentionlw(channel_out, channel_out)
        self.LECAlv = LECAlv(channel_out) #空间信息熵注意力

        self.conv_high = conv_block(channels_high, channel_out, 3, 1, bn_act=True, padding=1)
        self.ca = ChannelWise(channel_out)
        self.calw = ChannelWiselw(channel_out)
        self.LECA1 = LECA1(channel_out)  # 通道信息熵注意力

        self.FRB = nn.Sequential(
            conv_block(2 * channels_high, channel_out, 1, 1, bn_act=True, padding=0),
            conv_block(channel_out, channel_out, 3, 1, bn_act=True, group=1, padding=1))

        self.classifier = nn.Sequential(
            conv_block(channel_out, channel_out, 3, 1, padding=1, group=1, bn_act=True),
            nn.Dropout(p=0.15),
            conv_block(channel_out, classes, 1, 1, padding=0, bn_act=False))
        self.apf = conv_block(channel_out, channel_out, 3, 1, padding=1, group=1, bn_act=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.active = nn.Sigmoid()
        self.ESRA1 = ESRA(3*channel_out)
        self.LECA2 = LECA(channel_out)
        self.QKV = QKV(channel_out,sparsity=0.2)

        self.codebook = Codebook(3*channel_out, 3*channel_out, beta=0.25)

        # 熵引导融合模块
        self.entropy_fusion = nn.Sequential(
            conv_block(2 * channel_out, channel_out, 3, 1, bn_act=True, padding=1),
            conv_block(channel_out, channel_out, 3, 1, bn_act=True, padding=1)
        )

    def forward(self, x_high, x_low):
        _, _, h, w = x_low.size()

        lat_low = self.lateral_low(x_low)

        high_up1 = F.interpolate(x_high, size=lat_low.size()[2:], mode='bilinear', align_corners=False)

        concate = torch.cat([lat_low, high_up1], 1)
        concate = self.FRB(concate)

        conv_high = self.conv_high(high_up1)
        conv_low = self.conv_low(lat_low)


        # sa = self.sa(concate)
        sa = self.salw(concate)
        # sa = self.LECAlv(concate)
        # ca = self.ca(concate)
        ca = self.calw(concate)

        mul1 = torch.mul(sa, conv_high)
        mul2 = torch.mul(ca, conv_low)

        att_out1 = mul1 + mul2

        # print(att_out1.size())

        # 信息熵引导
        # Step1: 计算lat_low和att_out1的熵图
        entropy_att = -att_out1.softmax(dim=1) * (att_out1.softmax(dim=1) + 1e-8).log()
        entropy_att = entropy_att.sum(dim=1, keepdim=True)  # [B,1,H,W]

        entropy_low = -conv_low.softmax(dim=1) * (conv_low.softmax(dim=1) + 1e-8).log()
        entropy_low = entropy_low.sum(dim=1, keepdim=True)  # [B,1,H,W]

        # Step2: 熵图归一化作为权重
        weight_att = torch.sigmoid(entropy_att)
        weight_low = torch.sigmoid(entropy_low)

        # Step3: 熵加权融合
        fused = torch.cat([att_out1 * weight_att, conv_low * weight_low], dim=1)
        # print(fused.size())
        out = self.entropy_fusion(fused)

        sup = self.classifier(out)
        APF = self.apf(out)
        return APF,sup

class Codebook(nn.Module):
    def __init__(self, num_codebook_vectors,latent_dim,beta):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim = latent_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)
        # print("z_flattened.shape:",z_flattened.shape)
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))
        # print("d:",d.shape)
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        # print("embedding.weight:",self.embedding.weight.shape)
        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss
# ========== 空间注意力 (SpatialAttention) ==========
class SpatialAttentionlw(nn.Module):
    def __init__(self, in_ch, out_ch, droprate=0.15):
        super(SpatialAttentionlw, self).__init__()
        self.conv_sh = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.bn_sh1 = nn.BatchNorm2d(in_ch)
        self.bn_sh2 = nn.BatchNorm2d(in_ch)
        self.conv_res = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.drop = droprate
        self.fuse = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def entropy_spatial(self, x):
        """ 计算空间维度的信息熵 (B, C, H, W) → (B, 1, H, W) """
        epsilon = 1e-5
        p = x / (torch.sum(x, dim=1, keepdim=True) + epsilon)  # 归一化空间信息
        return -torch.sum(p * torch.log(p + epsilon), dim=1, keepdim=True)  # 计算熵

    def forward(self, x):
        b, c, h, w = x.size()

        # 计算空间信息熵
        spatial_entropy = self.entropy_spatial(x)  # (B, 1, H, W)
        entropy_weight = torch.sigmoid(spatial_entropy)  # 归一化熵值

        mxpool = F.max_pool2d(x, [h, 1])
        mxpool = self.bn_sh1(self.conv_sh(mxpool))

        avgpool = F.avg_pool2d(x, [h, 1])
        avgpool = self.bn_sh2(self.conv_sh(avgpool))

        att = torch.softmax(mxpool * avgpool, dim=1)

        attt1 = att[:, 0, :, :].unsqueeze(1)
        attt2 = att[:, 1, :, :].unsqueeze(1)

        fusion = attt1 * avgpool + attt2 * mxpool

        out = F.dropout(self.fuse(fusion), p=self.drop, training=self.training)

        # 信息熵调节注意力
        out = out * (1 + entropy_weight)

        out = F.relu(self.gamma * out + (1 - self.gamma) * x)

        return out


# ========== 通道注意力 (ChannelWise) ==========
class ChannelWiselw(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelWiselw, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False), nn.Sigmoid())

    def entropy_channel(self, x):
        """ 计算通道维度的信息熵 (B, C, H, W) → (B, C, 1, 1) """
        epsilon = 1e-5
        p = x / (torch.sum(x, dim=[2, 3], keepdim=True) + epsilon)  # 归一化通道信息
        return -torch.sum(p * torch.log(p + epsilon), dim=[2, 3], keepdim=True)  # 计算熵

    def forward(self, x):
        # 计算通道信息熵
        channel_entropy = self.entropy_channel(x)  # (B, C, 1, 1)
        entropy_weight = torch.sigmoid(channel_entropy)  # 归一化熵值

        y = self.avg_pool(x)
        y = self.conv_pool(y)

        # 信息熵调节注意力
        return x * y * (1 + entropy_weight)

class LCGB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        inner_ch = out_ch // 2
        self.down = NeighborDecouple(4) #4

        self.conv = nn.Sequential(
            conv_bn_relu(4 * in_ch, out_ch, 3, groups=4),
            conv_bn_relu(out_ch, out_ch, 1),
            conv_bn_relu(out_ch, out_ch, 3, groups=4),
            conv_bn_relu(out_ch, out_ch, 1),
            conv_bn_relu(out_ch, out_ch, 3, groups=4),
            conv_bn_relu(out_ch, out_ch, 1),
            conv_bn_relu(out_ch, out_ch, 3, groups=4),
            conv_bn_relu(out_ch, out_ch, 1),
        )

        self.conv1 = nn.Sequential(
            conv_bn_relu(16 * in_ch, out_ch, 3, groups=16),
            conv_bn_relu(out_ch, out_ch, 1),
            conv_bn_relu(out_ch, out_ch, 3, groups=16),
            conv_bn_relu(out_ch, out_ch, 1),
            conv_bn_relu(out_ch, out_ch, 3, groups=16),
            conv_bn_relu(out_ch, out_ch, 1),
            conv_bn_relu(out_ch, out_ch, 3, groups=16),
            conv_bn_relu(out_ch, out_ch, 1),
        )

        self.avgPool = nn.AvgPool2d(9, stride=8, padding=4, count_include_pad=False)
        self.to_qkv = conv_bn_relu(out_ch, 3 * inner_ch, 1)

        self.expand = conv_bn_relu(out_ch, out_ch, 1)

    def forward(self, x):
        # print(x.size())
        x = self.down(x)
        # print(x.size())
        x = self.conv1(x)
        # print(x.size())

        _, _, h, w = x.shape
        pooled = self.avgPool(x)

        qkv = self.to_qkv(pooled).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b c h w -> b c (h w)'), qkv)
        out = torch.einsum('bkl,bkt->blt', [k, q])
        out = F.softmax(out * (q.shape[1] ** (-0.5)), dim=1)
        out = torch.einsum('blt,btv->blv', [v, out])
        out = torch.cat([out, q], 1)
        out = rearrange(out, 'b c (h w) -> b c h w',
                        **parse_shape(pooled, 'b c h w'))
        out = self.expand(out)

        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        x = x + out

        return x


class NeighborDecouple(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        x = rearrange(x, 'b c (h h2) (w w2) -> b (h2 w2 c) h w', h2=self.bs, w2=self.bs)
        return x

class NeighborCouple(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        x = rearrange(x, 'b (h2 w2 c) h w -> b c (h h2) (w w2)', h2=self.bs, w2=self.bs)
        return x

class GlobalFeatureUpsample(nn.Module):
    def __init__(self, low_channels, in_channels, out_channels):
        super(GlobalFeatureUpsample, self).__init__()

        self.conv1 = conv_block(low_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=True)
        self.conv2 = nn.Sequential(
            conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=False),
            nn.ReLU(inplace=True))
        self.conv3 = conv_block(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=True)

    def forward(self, x_gui, y_high):
        h, w = x_gui.size(2), x_gui.size(3)
        y_up = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y_high)
        x_gui = self.conv1(x_gui)
        y_up = F.avg_pool2d(self.conv2(y_up), (1, 1))
        out = y_up + x_gui

        return self.conv3(out)



class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=(1, 1), group=1, bn_act=False,
                 bias=False):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=group, bias=bias)
        self.bn = SynchronizedBatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class SegHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegHead, self).__init__()

        self.fc = conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        return self.fc(x)


class SpatialAttentionLW(nn.Module):
    def __init__(self, in_ch, out_ch, droprate=0.15):
        super(SpatialAttentionLW, self).__init__()
        self.conv_sh = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.bn_sh1 = nn.BatchNorm2d(in_ch)
        self.bn_sh2 = nn.BatchNorm2d(in_ch)
        self.conv_res = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.drop = droprate
        self.fuse = conv_block(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def entropy_spatial(self, x):
        """ 计算空间维度的信息熵 (B, C, H, W) → (B, 1, H, W) """
        epsilon = 1e-5
        p = x / (torch.sum(x, dim=1, keepdim=True) + epsilon)  # 归一化空间信息
        return -torch.sum(p * torch.log(p + epsilon), dim=1, keepdim=True)  # 计算熵

    def forward(self, x):
        b, c, h, w = x.size()
        # print(x.size())
        # torch.Size([6, 256, 20, 20])

        # 计算空间信息熵
        spatial_entropy = entropy_spatial(x)  # (B, 1, H, W)
        entropy_weight = torch.sigmoid(spatial_entropy)  # 归一化熵值，使其在 (0,1) 之间

        mxpool = F.max_pool2d(x, [h, 1])  # .view(b,c,-1).permute(0,2,1)
        # print(mxpool.size())  #torch.Size([6, 256, 1, 20])
        # torch.Size([6, 256, 1, 20])
        # torch.Size([6, 128, 1, 40])
        # torch.Size([6, 64, 1, 80])
        # torch.Size([6, 32, 1, 160])
        mxpool = F.conv2d(mxpool, self.conv_sh.weight, padding=0, dilation=1)
        # torch.Size([6, 256, 1, 20])
        # torch.Size([6, 128, 1, 40])
        # torch.Size([6, 64, 1, 80])
        # torch.Size([6, 32, 1, 160])
        mxpool = self.bn_sh1(mxpool)
        # torch.Size([6, 256, 1, 20])
        # torch.Size([6, 128, 1, 40])
        # torch.Size([6, 64, 1, 80])
        # torch.Size([6, 32, 1, 160])

        avgpool = F.avg_pool2d(x, [h, 1])  # .view(b,c,-1)
        # torch.Size([6, 256, 1, 20])
        # torch.Size([6, 128, 1, 40])
        # torch.Size([6, 64, 1, 80])
        # torch.Size([6, 32, 1, 160])
        avgpool = F.conv2d(avgpool, self.conv_sh.weight, padding=0, dilation=1)
        avgpool = self.bn_sh2(avgpool)

        att = torch.softmax(torch.mul(mxpool, avgpool), 1)
        # torch.Size([6, 256, 1, 20])
        # torch.Size([6, 128, 1, 40])
        # torch.Size([6, 64, 1, 80])
        # torch.Size([6, 32, 1, 160])
        attt1 = att[:, 0, :, :].unsqueeze(1)
        # torch.Size([6, 1, 1, 20])
        # torch.Size([6, 1, 1, 40])
        # torch.Size([6, 1, 1, 80])
        # torch.Size([6, 1, 1, 160])
        attt2 = att[:, 1, :, :].unsqueeze(1)
        # print(attt2.size())
        # torch.Size([6, 1, 1, 20])
        # torch.Size([6, 1, 1, 40])
        # torch.Size([6, 1, 1, 80])
        # torch.Size([6, 1, 1, 160])

        fusion = attt1 * avgpool + attt2 * mxpool
        # print(fusion.size())
        # torch.Size([6, 256, 1, 20])
        # torch.Size([6, 128, 1, 40])
        # torch.Size([6, 64, 1, 80])
        # torch.Size([6, 32, 1, 160])

        out1 = F.dropout(self.fuse(fusion), p=self.drop, training=self.training)
        # print(out.size())
        # torch.Size([6, 256, 1, 20])
        # torch.Size([6, 128, 1, 40])
        # torch.Size([6, 64, 1, 80])
        # torch.Size([6, 32, 1, 160])
        # 信息熵调节注意力
        out = out1 * (1 + entropy_weight)
        # a1 = self.gamma * out
        # torch.Size([6, 256, 1, 20])
        # torch.Size([6, 128, 1, 40])
        # torch.Size([6, 64, 1, 80])
        # torch.Size([6, 32, 1, 160]
        # print(a1.size())
        # a = (1 - self.gamma) * x
        # print(a.size())
        # torch.Size([6, 256, 20, 20])
        # torch.Size([6, 128, 40, 40])
        # torch.Size([6, 64, 80, 80])
        # torch.Size([6, 32, 160, 160])

        # out = out.expand(residual.shape[0],residual.shape[1],residual.shape[2],residual.shape[3])
        out = F.relu(self.gamma * out + (1 - self.gamma) * x)
        # print(out.size())
        return out


class ChannelWiseLW(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelWiseLW, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Sequential(
            conv_block(channel, channel // reduction, 1, 1, padding=0, bias=False), nn.ReLU(inplace=False),
            conv_block(channel // reduction, channel, 1, 1, padding=0, bias=False), nn.Sigmoid())

        def entropy_channel(self, x):
            """ 计算通道维度的信息熵 (B, C, H, W) → (B, C, 1, 1) """
            epsilon = 1e-5
            p = x / (torch.sum(x, dim=[2, 3], keepdim=True) + epsilon)  # 归一化通道信息
            return -torch.sum(p * torch.log(p + epsilon), dim=[2, 3], keepdim=True)  # 计算熵

    def forward(self, x):
        # 计算通道信息熵
        channel_entropy = self.entropy_channel(x)  # (B, C, 1, 1)
        entropy_weight = torch.sigmoid(channel_entropy)  # 归一化熵值

        y = self.avg_pool(x)
        y = self.conv_pool(y)

        # return x * y
        # 信息熵调节注意力
        return x * y * (1 + entropy_weight)




class SpatialAttention(nn.Module):
    def __init__(self, in_ch, out_ch, droprate=0.15):
        super(SpatialAttention, self).__init__()
        self.conv_sh = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.bn_sh1 = nn.BatchNorm2d(in_ch)
        self.bn_sh2 = nn.BatchNorm2d(in_ch)
        self.conv_res = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.drop = droprate
        self.fuse = conv_block(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        # print(x.size())
        # torch.Size([6, 256, 20, 20])

        mxpool = F.max_pool2d(x, [h, 1])  # .view(b,c,-1).permute(0,2,1)
        # print(mxpool.size())  #torch.Size([6, 256, 1, 20])
        # torch.Size([6, 256, 1, 20])
        # torch.Size([6, 128, 1, 40])
        # torch.Size([6, 64, 1, 80])
        # torch.Size([6, 32, 1, 160])
        mxpool = F.conv2d(mxpool, self.conv_sh.weight, padding=0, dilation=1)
        # torch.Size([6, 256, 1, 20])
        # torch.Size([6, 128, 1, 40])
        # torch.Size([6, 64, 1, 80])
        # torch.Size([6, 32, 1, 160])
        mxpool = self.bn_sh1(mxpool)
        # torch.Size([6, 256, 1, 20])
        # torch.Size([6, 128, 1, 40])
        # torch.Size([6, 64, 1, 80])
        # torch.Size([6, 32, 1, 160])

        avgpool = F.avg_pool2d(x, [h, 1])  # .view(b,c,-1)
        # torch.Size([6, 256, 1, 20])
        # torch.Size([6, 128, 1, 40])
        # torch.Size([6, 64, 1, 80])
        # torch.Size([6, 32, 1, 160])
        avgpool = F.conv2d(avgpool, self.conv_sh.weight, padding=0, dilation=1)
        avgpool = self.bn_sh2(avgpool)

        att = torch.softmax(torch.mul(mxpool, avgpool), 1)
        # torch.Size([6, 256, 1, 20])
        # torch.Size([6, 128, 1, 40])
        # torch.Size([6, 64, 1, 80])
        # torch.Size([6, 32, 1, 160])
        attt1 = att[:, 0, :, :].unsqueeze(1)
        # torch.Size([6, 1, 1, 20])
        # torch.Size([6, 1, 1, 40])
        # torch.Size([6, 1, 1, 80])
        # torch.Size([6, 1, 1, 160])
        attt2 = att[:, 1, :, :].unsqueeze(1)
        # print(attt2.size())
        # torch.Size([6, 1, 1, 20])
        # torch.Size([6, 1, 1, 40])
        # torch.Size([6, 1, 1, 80])
        # torch.Size([6, 1, 1, 160])

        fusion = attt1 * avgpool + attt2 * mxpool
        # print(fusion.size())
        # torch.Size([6, 256, 1, 20])
        # torch.Size([6, 128, 1, 40])
        # torch.Size([6, 64, 1, 80])
        # torch.Size([6, 32, 1, 160])

        out = F.dropout(self.fuse(fusion), p=self.drop, training=self.training)
        # print(out.size())
        # torch.Size([6, 256, 1, 20])
        # torch.Size([6, 128, 1, 40])
        # torch.Size([6, 64, 1, 80])
        # torch.Size([6, 32, 1, 160])

        a1 = self.gamma * out
        # torch.Size([6, 256, 1, 20])
        # torch.Size([6, 128, 1, 40])
        # torch.Size([6, 64, 1, 80])
        # torch.Size([6, 32, 1, 160]
        # print(a1.size())
        a = (1 - self.gamma) * x
        # print(a.size())
        # torch.Size([6, 256, 20, 20])
        # torch.Size([6, 128, 40, 40])
        # torch.Size([6, 64, 80, 80])
        # torch.Size([6, 32, 160, 160])

        # out = out.expand(residual.shape[0],residual.shape[1],residual.shape[2],residual.shape[3])
        out = F.relu(self.gamma * out + (1 - self.gamma) * x)
        # print(out.size())
        return out


class ChannelWise(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelWise, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Sequential(
            conv_block(channel, channel // reduction, 1, 1, padding=0, bias=False), nn.ReLU(inplace=False),
            conv_block(channel // reduction, channel, 1, 1, padding=0, bias=False), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_pool(y)

        return x * y

class StripAttentionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(StripAttentionModule, self).__init__()
        self.conv1 = ConvBNReLU(in_chan, 64, ks=1, stride=1, padding=0)
        self.conv2 = ConvBNReLU(in_chan, 64, ks=1, stride=1, padding=0)
        self.conv3 = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        self.DecoderBlock1 = DecoderBlock1(64, out_chan, BatchNorm=nn.BatchNorm2d)#
        self.DecoderBlock2 = DecoderBlock2(64, out_chan, BatchNorm=nn.BatchNorm2d)#self.out_channels[4], self.out_channels[4], BatchNorm

        self.init_weight()

    def forward(self, x):
        q = self.conv1(x)
        q = self.DecoderBlock1(q) #
        batchsize, c_middle, h, w = q.size()
        q = F.avg_pool2d(q, [h, 1])
        q = q.view(batchsize, c_middle, -1).permute(0, 2, 1)

        k = self.conv2(x)
        # k = self.DecoderBlock2(k) #
        k = k.view(batchsize, c_middle, -1)
        attention_map = torch.bmm(q, k)
        attention_map = self.softmax(attention_map)

        v = self.conv3(x)
        # v = self.DecoderBlock2(v)
        c_out = v.size()[1]
        v = F.avg_pool2d(v, [h, 1])
        v = v.view(batchsize, c_out, -1)

        augmented_feature_map = torch.bmm(v, attention_map)
        augmented_feature_map = augmented_feature_map.view(batchsize, c_out, h, w)
        out = x + augmented_feature_map
        # out = augmented_feature_map
        return out
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class StripAttentionModule1(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(StripAttentionModule1, self).__init__()
        self.conv1 = ConvBNReLU(in_chan, 64, ks=1, stride=1, padding=0)
        self.conv2 = ConvBNReLU(in_chan, 64, ks=1, stride=1, padding=0)
        self.conv3 = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        self.DecoderBlock1 = DecoderBlock1(64, 64, BatchNorm=nn.BatchNorm2d)#
        self.DecoderBlock2 = DecoderBlock2(64, 64, BatchNorm=nn.BatchNorm2d)#self.out_channels[4], self.out_channels[4], BatchNorm

        self.init_weight()

    def forward(self, x):
        q = self.conv1(x)
        q = self.DecoderBlock1(q) #
        batchsize, c_middle, h, w = q.size()
        q = F.avg_pool2d(q, [h, 1])
        q = q.view(batchsize, c_middle, -1).permute(0, 2, 1)

        k = self.conv2(x)
        # k = self.DecoderBlock2(k) #
        k = k.view(batchsize, c_middle, -1)
        attention_map = torch.bmm(q, k)
        attention_map = self.softmax(attention_map)

        v = self.conv3(x)
        # v = self.DecoderBlock2(v)
        c_out = v.size()[1]
        v = F.avg_pool2d(v, [h, 1])
        v = v.view(batchsize, c_out, -1)

        augmented_feature_map = torch.bmm(v, attention_map)
        augmented_feature_map = augmented_feature_map.view(batchsize, c_out, h, w)
        out = x + augmented_feature_map
        # out = augmented_feature_map
        return out
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
# 修改
class StripAttentionModule2(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(StripAttentionModule2, self).__init__()
        self.conv1 = ConvBNReLU(in_chan, 64, ks=1, stride=1, padding=0)
        self.conv2 = ConvBNReLU(in_chan, 64, ks=1, stride=1, padding=0)
        self.conv3 = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        self.DecoderBlock1 = DecoderBlock1(64, 64, BatchNorm=nn.BatchNorm2d)#
        self.DecoderBlock2 = DecoderBlock2(64, 64, BatchNorm=nn.BatchNorm2d)#self.out_channels[4], self.out_channels[4], BatchNorm

        self.init_weight()

    def forward(self, x):
        q = self.conv1(x)
        # q = self.DecoderBlock1(q) #
        batchsize, c_middle, h, w = q.size()
        q = F.avg_pool2d(q, [h, 1])
        q = q.view(batchsize, c_middle, -1).permute(0, 2, 1)

        k = self.conv2(x)
        k = self.DecoderBlock2(k) #
        k = k.view(batchsize, c_middle, -1)
        attention_map = torch.bmm(q, k)
        attention_map = self.softmax(attention_map)

        v = self.conv3(x)
        # v = self.DecoderBlock2(v)
        c_out = v.size()[1]
        v = F.avg_pool2d(v, [h, 1])
        v = v.view(batchsize, c_out, -1)

        augmented_feature_map = torch.bmm(v, attention_map)
        augmented_feature_map = augmented_feature_map.view(batchsize, c_out, h, w)
        out = x + augmented_feature_map
        # out = augmented_feature_map
        return out
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

# 全局
class DecoderBlock2(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super(DecoderBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp

        # self.deconv1 = nn.Conv2d(
        #     in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        # )
        # self.deconv2 = nn.Conv2d(
        #     in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        # )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )

        self.bn2 = BatchNorm(in_channels // 8 + in_channels // 8)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels // 8 + in_channels // 8, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x, inp = False):
        # print(x.size())  #torch.Size([6, 512, 20, 20])
        # print("wwwwwwwww")
        x = self.conv1(x) #torch.Size([6, 128, 20, 20])

        x = self.bn1(x)  #torch.Size([6, 128, 20, 20])

        x = self.relu1(x)#torch.Size([6, 128, 20, 20])


        # x1 = self.deconv1(x)  #torch.Size([6, 64, 20, 20])
        #
        # x2 = self.deconv2(x)  #torch.Size([6, 64, 20, 20])

        # x31 = self.h_transform(x) #torch.Size([6, 128, 20, 39])
        # print(x31.size())
        # x32 = self.deconv3(x31) #torch.Size([6, 64, 20, 39])
        # print(x32.size())
        # x33 = self.inv_h_transform(x32) #torch.Size([6, 64, 20, 20])
        # print(x33.size())

        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x3, x4), 1)
        if self.inp:
            x = F.interpolate(x, scale_factor=2)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size() #torch.Size([6, 128, 20, 20])

        x = torch.nn.functional.pad(x, (0, shape[-1]))   #torch.Size([6, 128, 20, 40])

        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]] # torch.Size([6, 128, 780])

        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1) #torch.Size([6, 128, 20, 39])

        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)
# 全局
class DecoderBlock1(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super(DecoderBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp

        self.deconv1 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        # self.deconv3 = nn.Conv2d(
        #     in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        # )
        # self.deconv4 = nn.Conv2d(
        #     in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        # )

        self.bn2 = BatchNorm(in_channels // 8 + in_channels // 8)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels // 8 + in_channels // 8, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x, inp = False):
        # print(x.size())  #torch.Size([6, 512, 20, 20])
        # print("wwwwwwwww")
        x = self.conv1(x) #torch.Size([6, 128, 20, 20])

        x = self.bn1(x)  #torch.Size([6, 128, 20, 20])

        x = self.relu1(x)#torch.Size([6, 128, 20, 20])


        x1 = self.deconv1(x)  #torch.Size([6, 64, 20, 20])

        x2 = self.deconv2(x)  #torch.Size([6, 64, 20, 20])

        # x31 = self.h_transform(x) #torch.Size([6, 128, 20, 39])
        # print(x31.size())
        # x32 = self.deconv3(x31) #torch.Size([6, 64, 20, 39])
        # print(x32.size())
        # x33 = self.inv_h_transform(x32) #torch.Size([6, 64, 20, 20])
        # print(x33.size())

        # x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        # x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2), 1)
        if self.inp:
            x = F.interpolate(x, scale_factor=2)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size() #torch.Size([6, 128, 20, 20])

        x = torch.nn.functional.pad(x, (0, shape[-1]))   #torch.Size([6, 128, 20, 40])

        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]] # torch.Size([6, 128, 780])

        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1) #torch.Size([6, 128, 20, 39])

        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)



# 全局
# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
#         super(DecoderBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
#         self.bn1 = BatchNorm(in_channels // 4)
#         self.relu1 = nn.ReLU()
#         self.inp = inp
#
#         self.deconv1 = nn.Conv2d(
#             in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
#         )
#         self.deconv2 = nn.Conv2d(
#             in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
#         )
#         self.deconv3 = nn.Conv2d(
#             in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
#         )
#         self.deconv4 = nn.Conv2d(
#             in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
#         )
#
#         self.bn2 = BatchNorm(in_channels // 4 + in_channels // 4)
#         self.relu2 = nn.ReLU()
#         self.conv3 = nn.Conv2d(
#             in_channels // 4 + in_channels // 4, n_filters, 1)
#         self.bn3 = BatchNorm(n_filters)
#         self.relu3 = nn.ReLU()
#
#         self._init_weight()
#
#     def forward(self, x, inp = False):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#
#         x1 = self.deconv1(x)
#         x2 = self.deconv2(x)
#         x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
#         x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
#         # x = torch.cat((x1, x2, x3, x4), 1)
#         x5 = x3 + x4
#         x6 = x1 + x2
#
#         x5_flat = x5.view(x5.size(0), x5.size(1), -1)
#         x6_flat = x6.view(x6.size(0), x6.size(1), -1)
#
#         attn = torch.bmm(x5_flat.permute(0, 2, 1), x6_flat)
#         attn = F.softmax(attn, dim=-1)
#
#         # x_flat = x.view(x.size(0), x.size(1), -1)
#         x_flat = x.view(x.size(0), x.size(1), -1)
#         # Adjust x_flat to match the dimensions of the attention matrix
#         if attn.size(1) != x_flat.size(1):
#             x_flat = F.interpolate(x_flat, size=(attn.size(1)), mode='linear')
#
#         direction_constraint = torch.bmm(attn, x_flat)
#         direction_constraint = direction_constraint.view_as(x)
#
#         x = direction_constraint + x
#         if self.inp:
#             x = F.interpolate(x, scale_factor=2)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu3(x)
#         return x
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.ConvTranspose2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, SynchronizedBatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def h_transform(self, x):
#         shape = x.size()
#         x = torch.nn.functional.pad(x, (0, shape[-1]))
#         x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
#         x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
#         return x
#
#     def inv_h_transform(self, x):
#         shape = x.size()
#         x = x.reshape(shape[0], shape[1], -1).contiguous()
#         x = torch.nn.functional.pad(x, (0, shape[-2]))
#         x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
#         x = x[..., 0: shape[-2]]
#         return x
#
#     def v_transform(self, x):
#         x = x.permute(0, 1, 3, 2)
#         shape = x.size()
#         x = torch.nn.functional.pad(x, (0, shape[-1]))
#         x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
#         x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
#         return x.permute(0, 1, 3, 2)
#
#     def inv_v_transform(self, x):
#         x = x.permute(0, 1, 3, 2)
#         shape = x.size()
#         x = x.reshape(shape[0], shape[1], -1)
#         x = torch.nn.functional.pad(x, (0, shape[-2]))
#         x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
#         x = x[..., 0: shape[-2]]
#         return x.permute(0, 1, 3, 2)
#

#局部精细
class eca_layer(nn.Module):
    def __init__(self, channel, k_size):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x

class SENet_Block(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SENet_Block, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class MaskPredictor(nn.Module):
    def __init__(self, in_channels, wn=lambda x: torch.nn.utils.weight_norm(x)):
        super(MaskPredictor, self).__init__()
        self.spatial_mask = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1, bias=False)

    def forward(self, x):
        spa_mask = self.spatial_mask(x)
        spa_mask = F.gumbel_softmax(spa_mask, tau=1, hard=True, dim=1)
        return spa_mask


class RIFU(nn.Module):
    def __init__(self, n_feats, reduction=8, wn=lambda x: torch.nn.utils.weight_norm(x)):
        super(RIFU, self).__init__()
        self.CA = eca_layer(n_feats, k_size=3)
        self.SE = SENet_Block(n_feats, reduction=reduction)  #

        self.MaskPredictor = MaskPredictor(n_feats * 8 // 8)

        self.k = nn.Sequential(
            wn(nn.Conv2d(n_feats * 8 // 8, n_feats * 8 // 8, kernel_size=3, padding=1, stride=1, groups=1)),
            nn.LeakyReLU(0.05),
            )

        self.k1 = nn.Sequential(
            wn(nn.Conv2d(n_feats * 8 // 8, n_feats * 8 // 8, kernel_size=3, padding=1, stride=1, groups=1)),
            nn.LeakyReLU(0.05),
            )

        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x):
        res = x
        x = self.k(x)

        MaskPredictor = self.MaskPredictor(x)
        mask = (MaskPredictor[:, 1, ...]).unsqueeze(1)
        x = x * (mask.expand_as(x))

        x1 = self.k1(x)
        x2 = self.CA(x1)
        print(x2.size)
        x3 = self.SE(x1)
        print(x3.size())
        print("LLLLLLL")
        out = self.x_scale(x2) + self.res_scale(res)

        return out

class CIAM(nn.Module):
    def __init__(self, n_feats, wn=lambda x: torch.nn.utils.weight_norm(x)):
        super(CIAM, self).__init__()
        pooling_r = 2
        med_feats = n_feats // 1
        self.k1 = nn.Sequential(
            nn.ConvTranspose2d(n_feats, n_feats * 4 // 3, kernel_size=pooling_r, stride=pooling_r, padding=0, groups=1,
                               bias=True),
            nn.LeakyReLU(0.05),
            nn.Conv2d(n_feats * 4 // 3, n_feats, kernel_size=1, stride=2, padding=0, groups=1),
            )

        self.sig = nn.Sigmoid()

        self.k3 = RIFU(n_feats)

        self.k4 = RIFU(n_feats)

        self.k5 = RIFU(n_feats)

        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x):
        identity = x
        _, _, H, W = identity.shape
        x1_1 = self.k3(x)
        x1 = self.k4(x1_1)

        x1_s = self.sig(self.k1(x) + x)
        x1 = self.k5(x1_s * x1)

        out = self.res_scale(x1) + self.x_scale(identity)

        return out

if __name__ == "__main__":
    input1 = torch.rand(2, 3, 360, 480)
    model = SSFPN("resnet18",ResNet34M=False)
    summary(model, torch.rand((2, 3, 360, 480)))

    # python train_cityscapes.py --dataset camvid --model MSFFNet --batch_size 4 --max_epochs 300 --train_type trainval --lr 1e-3
