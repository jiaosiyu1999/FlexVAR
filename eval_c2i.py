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
from utils.evaluations.c2i.img2npy import create_npz_from_sample_folder
import torch.distributed as dist
import argparse
import torch.nn.functional as F




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--var_ckpt", type=str, default="out/VARd20/40epo/ar-ckpt-epo35.pth")
    parser.add_argument("--vae_ckpt", type=str, default="pretrained/FlexVAE.pth")
    parser.add_argument("--cfg", type=float, default=4, )
    parser.add_argument("--top_k", type=int, default=900, )
    parser.add_argument("--maxpn", type=int, default=16, )
    parser.add_argument("--depth", type=int, default=24, )
    parser.add_argument("--infer_patch_nums", type=str, default="1_2_3_4_5_6_8_10_13_16", )


    args = parser.parse_args()

    MODEL_DEPTH = args.depth    # TODO: =====> please specify MODEL_DEPTH <=====
    assert MODEL_DEPTH in {16, 20, 24, 30}
    infer_patch_nums = tuple(map(int, args.infer_patch_nums.replace('-', '_').split('_')))

    dist.init_process_group(backend='nccl')
    global_rank = dist.get_rank()
    device = global_rank % torch.cuda.device_count()
    if global_rank == 0:
        print(args)

    seed = 0 * dist.get_world_size() + global_rank
    # torch.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    torch.cuda.set_device(device)


    # download checkpoint
    vae_ckpt = args.vae_ckpt
    var_ckpt = args.var_ckpt

    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae, var = build_vae_var(
        V=8912,  # 8912 8192
        Cvae=32, device=device, num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
        vae_ckpt = vae_ckpt,
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
    var.update_patch_related(infer_patch_nums)
    print(f'prepare finished.')


    ############################# 2. Sample with classifier-free guidance

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    data_list = [i for i in range(1000)] * 50
    random.shuffle(data_list)
    length = len(data_list) // dist.get_world_size()
    data_sublist = data_list[global_rank::dist.get_world_size()]
    dataset = [data_sublist[i:i + args.batch_size] for i in range(0, len(data_sublist), args.batch_size)]

    save_root = f'vis/eval_c2i/d{MODEL_DEPTH}_cfg{args.cfg}_shape{infer_patch_nums[-1]*16}_{len(infer_patch_nums)}step_maxpn{args.maxpn}'
    os.makedirs(save_root, exist_ok=True)
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster

            for num, c in enumerate(dataset):
                label_B: torch.LongTensor = torch.tensor(c, device=device)

                t1 = time.time()
                # sample
                B = len(c)
                recon_B3HW = var.autoregressive_infer_cfg(vqvae = vae, B=B, label_B=label_B, infer_patch_nums=infer_patch_nums, 
                                                          cfg=args.cfg, top_k=args.top_k, top_p=0.95, g_seed=None, 
                                                          more_smooth=False, max_pn = args.maxpn)

                for i in range(len(c)):
                    save_image(recon_B3HW[i].unsqueeze(0), f"{save_root}/rank{global_rank}-{num * args.batch_size + i}.png", normalize=True, value_range=(-1, 1))
                t2 = time.time()

                if num % 5 == 0:
                    print(f"curr step {num}, total-step{len(dataset)}, processing time: {t2-t1}s, etc. {(t2-t1)*(len(dataset)-num)}")



    if global_rank == 0:
        time.sleep(1*60)
        create_npz_from_sample_folder(save_root)

        print('done !')
        print("save_root:", save_root)
        