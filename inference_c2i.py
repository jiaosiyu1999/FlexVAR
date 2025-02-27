import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import  build_vae_var
from torchvision.utils import save_image
import time

MODEL_DEPTH = 24    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}


# download checkpoint
var_ckpt = "pretrained/FlexVARd24-epo349.pth"
vae_ckpt = "pretrained/FlexVAE.pth"
# build vae, var
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=8912, Cvae=32,        # hard-coded VQVAE hyperparameters
        device=device, num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
        vae_ckpt = vae_ckpt, 
        flash_if_available=False, fused_if_available=False,
    )

# load checkpoints
ckpt = torch.load(var_ckpt, map_location='cpu')
if 'trainer' in ckpt.keys():
    ckpt = ckpt['trainer']['var_wo_ddp']
old_params = var.state_dict()
ckpt["attn_bias_for_masking"] = old_params["attn_bias_for_masking"]
var.load_state_dict(ckpt, strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
infer_patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16, )
print(f'prepare finished.')


############################# 2. Sample with classifier-free guidance

# set args
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
cfg = 4 #@param {type:"slider", min:1, max:10, step:0.1}
# class_labels = (0,1)  #@param {type:"raw"}
class_labels = (888, 9, 154, 11, 108, 293, 949, 39)  #@param {type:"raw"}
more_smooth = False # True for more smooth output

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

t1 = time.time()
maxpn = 16
# sample
B = len(class_labels)
label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
with torch.inference_mode():
    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
        recon_B3HW = var.autoregressive_infer_cfg(vqvae = vae, B=B, label_B=label_B, infer_patch_nums=infer_patch_nums, cfg=cfg, 
                                                  top_k=900, top_p=0.95, g_seed=seed, 
                                                  more_smooth=more_smooth, max_pn = maxpn, used_llamagen_cfg=True)
t2 = time.time()

save_path = f"vis/c2i/d{MODEL_DEPTH}--shape{infer_patch_nums[-1]*16}_{len(infer_patch_nums)}step_maxpn{maxpn}.png"
print(f"processing time: {t2-t1}s, image shape: {recon_B3HW.shape}, save_path: {save_path}")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
save_image(recon_B3HW, save_path, nrow=4, normalize=True, value_range=(-1, 1))
