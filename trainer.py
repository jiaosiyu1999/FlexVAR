import time
from typing import List, Optional, Tuple, Union

import torch, random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import FlexVAR
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class VARTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local, var_wo_ddp: FlexVAR, var: DDP,
        var_opt: AmpOptimizer, label_smooth: float,
    ):
        super(VARTrainer, self).__init__()
        
        self.var, self.vae_local = var, vae_local
        self.var_wo_ddp: FlexVAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt
        
        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L
        
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn*pn
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        self.vae_embedding = F.normalize(self.vae_local.quantize.embedding.weight, p=2, dim=-1).to(device)
        self.length = self.vae_embedding.shape[0] - 1
        self.Ch = self.vae_embedding.shape[1]

    def encode_var_wo_firstL(self, x, curr_patch_nums):
        h = self.vae_local.encode_conti(x)
        # return quant_z.reshape(quant_z.shape[0], quant_z.shape[1], -1), indices
        all_indices = []
        all_quant = []
        end = len(curr_patch_nums) -1
        for num in range(len(curr_patch_nums)):
            curr_hw = curr_patch_nums[num]
            _h = F.interpolate(h.clone(), size=(curr_hw, curr_hw), mode='area')
            quant, _, log = self.vae_local.quantize(_h)
            indices = log[-1].view(quant.shape[0], -1)
            all_indices.append(indices)
            if not num == end:
                next_hw = curr_patch_nums[num+1]
                
                next_quant = F.interpolate(quant, size=(next_hw, next_hw), mode='bicubic')
                next_quant = next_quant.reshape(quant.shape[0], quant.shape[1], -1)
                all_quant.append(next_quant)

        all_quant = torch.cat(all_quant, dim = 2).permute(0,2,1)
        all_indices = torch.cat(all_indices, dim = 1)
        # if random.random() < 0.1:
        #     bs, length = all_indices.shape
        #     random_ind = torch.randint(low=0, high=self.length, size=(bs, length//20), dtype=torch.int64)
        #     random_quant = self.vae_embedding[random_ind]
        #     random_indices = torch.randint(low=0, high=all_quant.shape[1]-1, size=(bs, length//20)).to(random_quant.device)

        #     index = random_indices.unsqueeze(-1).expand(-1, -1, self.Ch)
        #     all_quant.scatter_(1, index, random_quant)
        return all_quant, all_indices


    def gen_curr_patch_nums(self, ):
        if random.random() < 0.05:
            curr_patch_nums = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
        else:
            random_numbers = random.sample(range(3, 11), 5) + random.sample(range(11, 16), 2)
            # random_numbers = random.sample(range(2, 16), 8) 
            random_numbers.sort()
            curr_patch_nums = [1, 2] + random_numbers + [16]

        # drop scales    
        x = random.random()
        if x > 0.9:
            drop_index = random.choice(range(2, len(curr_patch_nums) - 1))
            curr_patch_nums.pop(drop_index)
        if x > 0.95:
            drop_index = random.choice(range(2, len(curr_patch_nums) - 1))
            curr_patch_nums.pop(drop_index)
        if x > 0.98:
            drop_index = random.choice(range(2, len(curr_patch_nums) - 1))
            curr_patch_nums.pop(drop_index)
        if x > 0.99:
            drop_index = random.choice(range(2, len(curr_patch_nums) - 1))
            curr_patch_nums.pop(drop_index)

        total_lens = sum(pn ** 2 for pn in curr_patch_nums)
        while total_lens > 680:
            drop_index = random.choice(range(len(curr_patch_nums)-4, len(curr_patch_nums)))
            curr_patch_nums.pop(drop_index)
            total_lens = sum(pn ** 2 for pn in curr_patch_nums)
        return curr_patch_nums

    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # if progressive training
        # self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        # if self.last_prog_si != prog_si:
        #     if self.last_prog_si != -1: self.first_prog = False
        #     self.last_prog_si = prog_si
        #     self.prog_it = 0
        # self.prog_it += 1
        # prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        # if self.first_prog: prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
        # if prog_si == len(self.patch_nums) - 1: prog_si = -1    # max prog, as if no prog

        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping
        curr_patch_nums = self.gen_curr_patch_nums()
        quant_z, gt_BL = self.encode_var_wo_firstL(inp_B3HW, curr_patch_nums)
        
        
        with self.var_opt.amp_ctx:
            self.var_wo_ddp.forward
            logits_BLV = self.var(label_B, quant_z, curr_patch_nums)
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            # if prog_si >= 0:    # in progressive training
            #     bg, ed = self.begin_ends[prog_si]
            #     assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
            #     lw = self.loss_weight[:, :ed].clone()
            #     lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            # else:               # not in progressive training
            #     lw = self.loss_weight
            # loss = loss.mul(lw).sum(dim=-1).mean()

            L = sum(pn * pn for pn in curr_patch_nums)
            lw = torch.ones(1, L, device=self.loss_weight.device) / L
            
            loss = loss.mul(lw).sum(dim=-1).mean()

        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it % metric_lg.log_iters == 0:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            if prog_si >= 0:    # in progressive training
                Ltail = acc_tail = -1
            else:               # not in progressive training
                Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
                acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
        
        return grad_norm, scale_log2
    
    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)
