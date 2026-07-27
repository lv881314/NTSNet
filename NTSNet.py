class GaussianFourierAttention(nn.Module):   ##GFAM
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
        xxxxxxxxxxxxxxxxxxxxxxxxxx
        xxxxxxxxxxxxxxxxxxxxxxxxxx
        xxxxxxxxxxxxxxxxxxxxxxxxxx

        return out


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


        xxxxxxxxxxxxxxxxxxxxxxxxxx
        xxxxxxxxxxxxxxxxxxxxxxxxxx
        xxxxxxxxxxxxxxxxxxxxxxxxxx
        return APF


