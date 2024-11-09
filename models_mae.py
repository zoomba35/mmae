# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np
import cv2

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)  # 切块 使用卷积实现 适应多尺度图像输入
        num_patches = self.patch_embed.num_patches  # patch的总数

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # cls-token初始化
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        # 位置编码器 （cls-token+196,768（embedding））初始化（全0）
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            # Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])  # 创建transformers 编码器
        self.norm = norm_layer(embed_dim)  # 规范化层
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim,
                                       bias=True)  # 解码器 （encoder-embedding）->（decoder-embedding） 全连接层

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))  # 解码器 掩码初始化。。。这个不太清楚

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # 解码器的位置编码器层 patches+1是为了匹配cls-token  fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            # Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])  # 解码器tf模块

        self.decoder_norm = norm_layer(decoder_embed_dim)  # 解码器规范化层
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # 解码器输出层使用全连接 这里应该是使用cls-token输出的  输入维度decoder-embedding，输出维度 patch（16x16x3）还原每个patch的每个通道内的像素
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()  # 权重初始化



    def initialize_weights(self):
        '''
        权重初始化，方法同vit权重初始化方法
        :return:
        '''
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)  # 获取2维位置编码 输入参数 embeding嵌入维度，int（根号下patch的数量，也就是宽度），是否计算cls-token
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))  # 复制一份

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5),
                                                    cls_token=True)  # 同上计算一份解码器的位置编码
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))  # 复制一份

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data  # 使用全连接层初始化方式进行数据初始化而不是使用vit中的卷积初始化 这里可能是认为卷积中的平移不变性存在先验知识
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))  # 均匀分布初始化

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)  # 正态分布初始化
        torch.nn.init.normal_(self.mask_token, std=.02)  # 正态分布初始化

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)  # 将初始化值赋值到当前模型中

    def _init_weights(self, m):
        '''
        vit权重初始化
        :param m:
        :return:
        '''
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT: 改进型的始化方法,从vit中的正太分布到均匀分布初始化 这里可能是认为卷积中的平移不变性存在先验知识
            torch.nn.init.xavier_uniform_(m.weight)  # 均匀分布初始化
            if isinstance(m, nn.Linear) and m.bias is not None:  # 如果参数属于线性层并且偏置不为0
                nn.init.constant_(m.bias, 0)  # 使用0将偏置向量进行填充
        elif isinstance(m, nn.LayerNorm):  # 如果参数属于规范化层
            nn.init.constant_(m.bias, 0)  # 偏置向量使用全零初始化
            nn.init.constant_(m.weight, 1.0)  # 权重向量使用全1初始化

    def patchify(self, imgs):
        """
        vit中切分patch调整数据维度的操作
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]  # 每个patch的长宽
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0  # 如果图像的长宽相等并且长宽可以整除patch的长度 才可以通过断言

        h = w = imgs.shape[2] // p  # 计算patch的数量
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        # batchsize，通道数，patch每一边的数量，每个patch的宽高，patch每一边的数量，每个patch的长宽
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        # 其实直接转换过来就行， （batchsize，patch总数base=196，path平方x通道数也就是每个patch块内的数据）
        return x

    def unpatchify(self, x):
        """
        反切分，拼接回来
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        # （batchsize，patch总数base=196，path平方x通道数也就是每个patch块内的数据）->（图像宽高，通道数，hxp=patch[0]xp的宽高）
        return imgs


    def combined_masking(self, x, mask_ratio):
        '''
        使用随机噪声进行mask掩码操作
        :param x:
        :param mask_ratio: 掩码率 也就是mask百分比
        :return:
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        '''
        N, L, D = x.shape  # batch, length 196, dim 768

        # Random Masking
        len_keep_random = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore_random = torch.argsort(ids_shuffle, dim=1)
        ids_keep_random = ids_shuffle[:, :len_keep_random]
        x_masked_random = torch.gather(x, dim=1, index=ids_keep_random.unsqueeze(-1).repeat(1, 1, D))
        mask_random = torch.ones([N, L], device=x.device)
        mask_random[:, :len_keep_random] = 0
        mask_random = torch.gather(mask_random, dim=1, index=ids_restore_random)

        # Patch Masking
        patch_avg = torch.mean(x, dim=2)
        ids_restore_patch = torch.argsort(patch_avg, dim=1)
        len_patch_keep = max(int((L * (1 - mask_ratio)) / 2), 1)
        ids_keep_patch = ids_restore_patch[:, :len_patch_keep]
        ids_keep_patch = torch.cat((ids_keep_patch, ids_restore_patch[:, -len_patch_keep:]), dim=1)
        x_masked_patch = torch.gather(x, dim=1, index=ids_keep_patch.unsqueeze(-1).repeat(1, 1, D))
        mask_patch = torch.ones([N, L], device=x.device)
        mask_patch.scatter_(1, ids_keep_patch, 0)



        match = torch.all(torch.isin(ids_keep_patch[:, 0], ids_keep_random))
        if match:
            ids_keep_random[:, 0] = ids_keep_patch[:, 0]
        match = torch.all(torch.isin(ids_keep_patch[:, -1], ids_keep_random))
        if match:
            ids_keep_random[:, -1] = ids_keep_patch[:, -1]

        x_masked_random = torch.gather(x, dim=1, index=ids_keep_random.unsqueeze(-1).repeat(1, 1, D))
        mask_random[:, :len_keep_random] = 0
        mask_random = torch.gather(mask_random, dim=1, index=ids_restore_random)
        x_masked_patch = torch.gather(x, dim=1, index=ids_keep_patch.unsqueeze(-1).repeat(1, 1, D))

#         print(ids_keep_patch)
#         print(ids_keep_random)

#         print(ids_keep_patch.shape)
#         print(ids_keep_random.shape)

        return x_masked_random, mask_random, ids_restore_random





    def forward_encoder(self, x, mask_ratio):
        '''
        编码器计算
        :param x: 图像
        :param mask_ratio: 掩码率
        :return:
        '''
        # embed patches
        x = self.patch_embed(x)  # 图像切分为patch的操作后reshape的数据

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]  # cls的添加也就是 (196,768)->(197,769)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.combined_masking(x, mask_ratio)  # 添加噪点的图像，掩码地方的像素，下标
        # x, mask_random, ids_restore_random = self.combined_masking(x, mask_ratio)  # 添加噪点的图像，掩码地方的像素，下标

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # 全0初始化后+embedding中第一个元素也就是cls对应的元素位置相加
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # cls-token进行形状调整，并在内存创建一个新的变量 batchsize，length，dim
        x = torch.cat((cls_tokens, x), dim=1)
        # concat拼接在一起，1x1xdim(768) concat batch_size x length x dim ）也就是在196维度上添加一个cls-token

        # apply Transformer blocks
        for blk in self.blocks:  # tf中的编码器层
            x = blk(x)
        x = self.norm(x)  # 规范化

        return x, mask, ids_restore
        # return x, mask_random, ids_restore_random


    # def forward_decoder(self, x, ids_restore):
    def forward_decoder(self, x, ids_restore_random):

        '''
        解码器前向传播计算
        :param x:
        :param ids_restore: 对应下标
        :return:
        '''
        # embed tokens
        x = self.decoder_embed(x)  # 线性层将编码器输出对齐 主要是将dim对齐

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore_random.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token  还原噪点图像
        x_ = torch.gather(x_, dim=1, index=ids_restore_random.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle 替换噪点值
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token 拼接cls-token

        # add pos embed
        x = x + self.decoder_pos_embed  # 添加位置编码

        # apply Transformer blocks
        for blk in self.decoder_blocks:  # 解码器的tf模块 规范化+多头自注意力+规范化+前馈全连接+残差
            x = blk(x)
        x = self.decoder_norm(x)  # 规范化

        # predictor projection
        x = self.decoder_pred(x)  # 还原全连接层 还原每个patch中的像素值 也就是batchsize X length X patch^2x3

        # remove cls token
        x = x[:, 1:, :]  # 删除cls-token

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)  # 目标值的形状调整 [N, 3, H, W]->[N, L, p*p*3]
        if self.norm_pix_loss:  # 如果计算归一化的loss
            mean = target.mean(dim=-1, keepdim=True)  # 计算均值
            var = target.var(dim=-1, keepdim=True)  # 计算样本方差
            target = (target - mean) / (var + 1.e-6) ** .5  # x-均值/方差开根 得到正态分布

        loss = (pred - target) ** 2  # 欧氏距离
        loss = loss.mean(dim=-1)  # 计算每块的像素点的均值 [N, L], mean loss per patch
        # print(mask.sum() )
        if mask.sum() == 0:
            print("error")
        loss = (loss * mask).sum() / mask.sum() # 仅仅只计算被遮盖住区域像素点误差的均值 也就是删除掉的块内像素点误差均值  mean loss on removed patches
        return loss



    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        # latent编码结果，掩码[batchsize,length(0是保存下来的batch，1是删除掉的batch)]，下标
        pred = self.forward_decoder(latent, ids_restore)  # 预测值 [N, L, p*p*3]

        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks


#     def random_masking(self, x, mask_ratio):
#         """
#         Perform per-sample random masking by per-sample shuffling.
#         Per-sample shuffling is done by argsort random noise.
#         x: [N, L, D], sequence
#         """
#         N, L, D = x.shape  # batch, length, dim
#         len_keep = int(L * (1 - mask_ratio))
#
#         noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
#
#         # sort noise for each sample
#         ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
#         ids_restore = torch.argsort(ids_shuffle, dim=1)
#
#         # keep the first subset
#         ids_keep = ids_shuffle[:, :len_keep]
#         x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
#
#         # generate the binary mask: 0 is keep, 1 is remove
#         mask = torch.ones([N, L], device=x.device)
#         mask[:, :len_keep] = 0
#         # unshuffle to get the binary mask
#         mask = torch.gather(mask, dim=1, index=ids_restore)
#
#         return x_masked, mask, ids_restore
#
#
#     def patch_masking(self, x, mask_ratio):
#         '''
#         使用随机噪声进行mask掩码操作
#         :param x:
#         :param mask_ratio: 掩码率 也就是mask百分比
#         :return:
#         Perform per-sample random masking by per-sample shuffling.
#         Per-sample shuffling is done by argsort random noise.
#         x: [N, L, D], sequence
#         '''
#         N, L, D = x.shape  # batch, length 196, dim 768
#         # len_keep = int(L * mask_ratio)  # 计算掩码率对应需要保存下来的length个数
#         patch_avg = torch.mean(x, dim=2)  # 计算每个patch的平均值
#         # ids_shuffle = torch.argsort(patch_avg, dim=1)  # 从小到大排序
#         ids_restore = torch.argsort(patch_avg, dim=1)
#
#         # keep the first 15% and last 15% patches
#         len_patch_keep = max(int((L * (1 - mask_ratio)) / 2), 1)  # 至少保留一个patch
#         ids_keep = ids_restore[:, :len_patch_keep]  # 获取需要保留的patch的下标
#         ids_keep = torch.cat((ids_keep, ids_restore[:, -len_patch_keep:]), dim=1)  # 将前15%和后15%的patch拼接起来
#
#         x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # 将对应下标元素进行位置的改变
#
#         # generate the binary mask: 0 is keep, 1 is remove
#         mask = torch.ones([N, L], device=x.device)  # 全1初始化batchsizex196
#         mask.scatter_(1, ids_keep, 0)  # 将需要保留的patch对应位置置为0，其余位置为1
#
#         return x_masked, mask, ids_restore
#
#
#
#
#     def forward_encoder(self, x, mask_ratio):
#         # embed patches
#
#         x = self.patch_embed(x)
#
#         # add pos embed w/o cls token
#         x = x + self.pos_embed[:, 1:, :]
#
#         # masking: length -> length * mask_ratio
#         x, mask, ids_restore = self.patch_masking(x, mask_ratio)
#
#         # append cls token
#         cls_token = self.cls_token + self.pos_embed[:, :1, :]
#         cls_tokens = cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#
#         # apply Transformer blocks
#         for blk in self.blocks:
#             x = blk(x)
#         x = self.norm(x)
#
#         return x, mask, ids_restore
#
#     def forward_decoder(self, x, ids_restore):
#         # embed tokens
#         x = self.decoder_embed(x)
#
#         # append mask tokens to sequence
#         mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
#         x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
#         x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
#         x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
#
#         # add pos embed
#         x = x + self.decoder_pos_embed
#
#         # apply Transformer blocks
#         for blk in self.decoder_blocks:
#             x = blk(x)
#         x = self.decoder_norm(x)
#
#         # predictor projection
#         x = self.decoder_pred(x)
#
#         # remove cls token
#         x = x[:, 1:, :]
#
#         return x
#
#     def forward_loss(self, imgs, pred, mask):
#         """
#         imgs: [N, 3, H, W]
#         pred: [N, L, p*p*3]
#         mask: [N, L], 0 is keep, 1 is remove,
#         """
#         target = self.patchify(imgs)
#         if self.norm_pix_loss:
#             mean = target.mean(dim=-1, keepdim=True)
#             var = target.var(dim=-1, keepdim=True)
#             target = (target - mean) / (var + 1.e-6)**.5
#
#         loss = (pred - target) ** 2
#         loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
#
#         loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
#         return loss
#
#     def forward(self, imgs, mask_ratio=0.75):
#         latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
#         pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
#         loss = self.forward_loss(imgs, pred, mask)
#         return loss, pred, mask
#
#
#
# def mae_vit_base_patch16_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
#
#
# def mae_vit_large_patch16_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
#
#
# def mae_vit_huge_patch14_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=14, embed_dim=1280, depth=32, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
#
#
# # set recommended archs
# mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks