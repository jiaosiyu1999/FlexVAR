import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_

try:
    from xformers.ops import memory_efficient_attention
    use_xformer = True
except ImportError:
    use_xformer = False


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class FlexVAR(nn.Module):
    def __init__(
        self, vae_local,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1, token_dropout_p=0.05,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = self.generate_2d_rotary_position_embedding(h=32, w=32, d = self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        self.tok_dropout = nn.Dropout(token_dropout_p)

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6))

    def get_num_params(self, ):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h, cond_BD))
        # return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    def update_patch_related(self, infer_patch_nums):
        self.patch_nums = infer_patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        cur = 0
        self.begin_ends = []
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        self.num_stages_minus_1 = len(self.patch_nums) - 1

        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        if use_xformer:
            b, c, l, l = attn_bias_for_masking.shape
            # l_ = 680
            l_ = (l + 7) // 8 * 8
            attn_bias_for_masking_ = torch.full((b, c, l_, l_), -torch.inf)
            attn_bias_for_masking_[:, :, :l, :l] = attn_bias_for_masking
            attn_bias_for_masking_[:,:,l:, l:] = 0
            attn_bias_for_masking = attn_bias_for_masking_

        self.attn_bias_for_masking = attn_bias_for_masking.to(self.attn_bias_for_masking.device)


    def generate_2d_rotary_position_embedding(self, h, w, d):
        assert d % 2 == 0, "Dimension d must be an even number."
        
        pos_encoding = torch.zeros(h, w, d)
        y_coords = torch.arange(h, dtype=torch.float32)
        x_coords = torch.arange(w, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords)
        
        div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * -(math.log(10000.0) / d))
        
        for i in range(h):
            for j in range(w):
                pos_encoding[i, j, 0::2] = torch.sin(y_grid[i, j] * div_term)
                pos_encoding[i, j, 1::2] = torch.cos(x_grid[i, j] * div_term)
        
        return pos_encoding.unsqueeze(0).permute(0,3,1,2)


    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, vqvae, B: int, label_B: Optional[Union[int, torch.LongTensor]], infer_patch_nums,
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False, max_pn = 16, used_llamagen_cfg=False, invalid_ids=None,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng

        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))

        pos_1LC = []
        for hw in infer_patch_nums:
            curr_pos_1LC = F.interpolate(self.pos_1LC, size=(hw, hw), mode='area')  # downsample 用 area
            curr_pos_1LC = curr_pos_1LC.reshape(1, self.pos_1LC.shape[1], -1).permute(0,2,1)
            pos_1LC.append(curr_pos_1LC)
        pos_1LC = torch.cat(pos_1LC, dim = 1)
        lvl_pos = pos_1LC

        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        vae_embedding = F.normalize(vqvae.quantize.embedding.weight, p=2, dim=-1)
        

        cur_L = 0
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(infer_patch_nums):   # si: i-th segment
            ratio = si / min(len(infer_patch_nums)-1, 9)
            # ratio = si / (len(infer_patch_nums) - 1)
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            
            if invalid_ids is not None:
                logits_BlV[:, :, invalid_ids] = -100.0

            if not used_llamagen_cfg:
                # # cfg-var
                t = cfg * ratio if pn <= max_pn else cfg
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            else:
                # cfg-llamagen
                # t = cfg * (0.5 + 0.5*ratio)
                cond_logits, uncond_logits = torch.split(logits_BlV, len(logits_BlV) // 2, dim=0)
                logits_BlV = uncond_logits + (cond_logits - uncond_logits) * cfg
            
            if pn > max_pn or pn not in [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 23, 24, 32]:
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=top_p, num_samples=1)[:, :, 0]
            else:
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

            # if not more_smooth: # this is the default case
            #     h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            # else:   # not used when evaluating FID/IS/Precision/Recall
            #     gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
            #     h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            # print(pn, torch.isin(idx_Bl,invalid_ids.to(idx_Bl.device)).any())
            
            assert not more_smooth # this is the default case
            if si != len(infer_patch_nums)-1:
                next_hw = infer_patch_nums[si+1]
                curr_hw = infer_patch_nums[si]
                
                curr_quant = vae_embedding[idx_Bl].reshape(B, curr_hw, curr_hw, vae_embedding.shape[-1]).permute(0,3,1,2)
                next_quant = F.interpolate(curr_quant, size=(next_hw, next_hw), mode='bicubic')
                next_quant = next_quant.reshape(B, curr_quant.shape[1], -1).permute(0,2,1)

                next_token_map = self.word_embed(next_quant) + lvl_pos[:, cur_L:cur_L + infer_patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        z_shape = [logits_BlV.shape[0], 32, infer_patch_nums[-1], infer_patch_nums[-1]]
        samples = vqvae.decode_code(idx_Bl, shape=z_shape)

        for b in self.blocks: b.attn.kv_caching(False)
        return samples
    
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor, infer_patch_nums) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        self.update_patch_related(infer_patch_nums)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.to(sos.dtype))), dim=1)

            pos_1LC = []
            for hw in self.patch_nums:
                curr_pos_1LC = F.interpolate(self.pos_1LC, size=(hw, hw), mode='area')
                curr_pos_1LC = curr_pos_1LC.reshape(1, self.pos_1LC.shape[1], -1).permute(0,2,1)
                pos_1LC.append(curr_pos_1LC)
            pos_1LC = torch.cat(pos_1LC, dim = 1)

            x_BLC += pos_1LC[:, :self.L] # lvl: BLC;  pos: 1LC
        
        attn_bias = self.attn_bias_for_masking
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC, cond_BD)
        # x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        # print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'

