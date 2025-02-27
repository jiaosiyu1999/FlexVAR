import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import torch, random
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy, size_based_auto_wrap_policy

import time
import argparse
import functools
import inspect
import contextlib
from dataset.build import build_dataset

import dist 
# import torch.distributed as dist

from utils import arg_util, misc
from utils.logger import create_logger
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler

# build models
from models import FlexVAR, build_vae_var


def custom_collate_fn(batch):
    return batch

def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace, device) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in list(model.blocks),
        ),

        # auto_wrap_policy=size_based_auto_wrap_policy,
        # process_group=fs_init.get_data_parallel_group(),
        device_id=device,
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
            "hsdp": ShardingStrategy.HYBRID_SHARD,
        }["fsdp"],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[args.mixed_precision],
            reduce_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[args.mixed_precision],
        ),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )

    torch.cuda.synchronize()

    return model



def creat_optimizer_by_name(model, weight_decay, learning_rate, betas, global_rank, logger):
    # start with all of the candidate parameters
    all_param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in all_param_dict.items() if p.requires_grad}
    
    # create optim groups. 
    # Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    
    # decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    # nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    # model params are flatten by fsdp, we need to set the params by its name
    decay_params = [p for n, p in param_dict.items() if 'norm' not in n]
    nodecay_params = [p for n, p in param_dict.items() if 'norm' in n]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"(rank {global_rank}) num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"(rank {global_rank}) num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    print(f"(rank {global_rank}) num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"(rank {global_rank}) num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer


def encode_var_wo_firstL(vae_local, x, curr_patch_nums):
    h = vae_local.encode_conti(x)
    # return quant_z.reshape(quant_z.shape[0], quant_z.shape[1], -1), indices
    all_indices = []
    all_quant = []
    end = len(curr_patch_nums) -1
    for num in range(len(curr_patch_nums)):
        curr_hw = curr_patch_nums[num]
        _h = F.interpolate(h.clone(), size=(curr_hw, curr_hw), mode='area')
        quant, _, log = vae_local.quantize(_h)
        indices = log[-1].view(quant.shape[0], -1)
        all_indices.append(indices)
        if not num == end:
            next_hw = curr_patch_nums[num+1]
            
            next_quant = F.interpolate(quant, size=(next_hw, next_hw), mode='bicubic')
            next_quant = next_quant.reshape(quant.shape[0], quant.shape[1], -1)
            all_quant.append(next_quant)

    all_quant = torch.cat(all_quant, dim = 2).permute(0,2,1)
    all_indices = torch.cat(all_indices, dim = 1)

    return all_quant, all_indices


def gen_curr_patch_nums():
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
    
if __name__ == "__main__":
    args = arg_util.init_dist_and_get_args()
    global_rank = dist.get_rank()
    device = global_rank % torch.cuda.device_count()
    seed = args.seed * dist.get_world_size() + global_rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={global_rank}, device={device}, seed={seed}, world_size={dist.get_world_size()}.")

    # =======================================
    #    Initialize logger and wandb
    # =======================================

    experiment_dir = args.tb_log_dir_path
    cloud_checkpoint_dir = f"{args.tb_log_dir_path}/ckpt"
    if global_rank == 0:
        os.makedirs(experiment_dir, exist_ok=True) # in each local machine
        os.makedirs(cloud_checkpoint_dir, exist_ok=True) # in one shared file storage
        logger = create_logger(experiment_dir)
    else:
        logger = create_logger(None)
    logger.info(f"Starting rank={global_rank}, device={device}, seed={seed}, world_size={dist.get_world_size()}.")
    logger.info(f"Experiment directory created at {experiment_dir}")
    logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")

    # training args
    logger.info(f"{args}")  

    if args.continue_training_ckpt:
        with open(os.path.join(args.continue_training_ckpt, "resume_step.txt")) as f:
            train_steps = int(f.read().strip())
        # start_epoch = int(train_steps / int(len(train_data) / global_batch_size))
        # train_steps = int(start_epoch * int(len(train_data) / global_batch_size))
        logger.info(f"Initial state: steps={train_steps}, epochs={train_steps} (need to change!!!!!!!!)")
    else:
        train_steps = 0
        start_epoch = 0

    # create dataset
    num_classes, train_data, _ = build_dataset(args)
    loader = DataLoader(
            dataset=train_data, num_workers=args.workers, pin_memory=True,
            generator=args.get_different_generator_for_each_rank(), # worker_init_fn=worker_init_fn,
            batch_sampler=DistInfiniteBatchSampler(
                dataset_len=len(train_data), glb_batch_size=args.glb_batch_size, same_seed_for_all_ranks=args.same_seed_for_all_ranks,
                shuffle=True, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(), start_ep=start_epoch, start_it=train_steps,
            ),
            # collate_fn=custom_collate_fn
    )

    logger.info(f"Dataset contains {len(train_data):,}  ")

    # create model 
    vae_local, var_wo_ddp = build_vae_var(
        V=8912,  # 8912 8192
        Cvae=32, ch=160,         # hard-coded VQVAE hyperparameters
        device=dist.get_device(), patch_nums=args.patch_nums,
        num_classes=num_classes, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm, token_dropout_p=args.token_dropout_p,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini, vae_ckpt=args.vae_ckpt,
    )
    
    vae_local = args.compile_model(vae_local, args.vfast)
    var_wo_ddp: FlexVAR = args.compile_model(var_wo_ddp, args.tfast)
    var_wo_ddp = var_wo_ddp.to(device)
    logger.info(f"FlexVAR Parameters: {(var_wo_ddp.get_num_params()/1e6)} M.")
    var_wo_ddp = setup_fsdp_sync(var_wo_ddp, args, device)


    
    
    # create optimizer
    optimizer = creat_optimizer_by_name(var_wo_ddp, args.twde, args.tblr, (0.9, 0.95), global_rank, logger)
    if args.continue_training_ckpt:
        opt_state_world_size = len([
            x for x in os.listdir(args.aim_ckpt)
            if x.startswith("optimizer.") and x.endswith(".pth")
        ])
        assert opt_state_world_size == dist.get_world_size(), (
            f"Resuming from a checkpoint with unmatched world size "
            f"({dist.get_world_size()} vs. {opt_state_world_size}) "
            f"is currently not supported."
        )
        logger.info(f"Resuming optimizer states from: {args.aim_ckpt}")
        optimizer.load_state_dict(torch.load(os.path.join(
            args.aim_ckpt,
            f"optimizer.{dist.get_rank():05d}-of-"
            f"{dist.get_world_size():05d}.pth",
        ), map_location="cpu"))


    # ======================================================
    #   Start training !!!
    # ======================================================
    train_loss = nn.CrossEntropyLoss(label_smoothing=args.ls, reduction='none')

    global_batch_size = int(args.batch_size) * dist.get_world_size()
    # assert dist.get_world_size() == 16, dist.get_world_size()



    var_wo_ddp.train()  # important! This enables embedding dropout for classifier-free guidance
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()
    total_steps = len(train_data) * (args.ep - start_epoch) / global_batch_size


    logger.info(f"Training for {args.ep} epochs...")
    for epoch in range(start_epoch, args.ep):
        # sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for inp, label in loader:

            inp = inp.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)

            B, V = label.shape[0], vae_local.vocab_size
            curr_patch_nums = gen_curr_patch_nums()
            quant_z, gt_BL = encode_var_wo_firstL(vae_local, inp, curr_patch_nums)


            optimizer.zero_grad()
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.mixed_precision]: 
                logits_BLV = var_wo_ddp(label, quant_z, curr_patch_nums)

                # t5=time.time()
                loss = train_loss(logits_BLV.reshape(-1, V), gt_BL.view(-1)).view(B, -1).mean()

                # _, loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices)
            loss.backward()
            
            var_wo_ddp.clip_grad_norm_(1.0)
            optimizer.step()
            

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.tdist.all_reduce(avg_loss, op=dist.tdist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Total steps: {total_steps}, etc. {(total_steps - train_steps) / steps_per_sec / 3600.0} h.")


                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()


            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                cloud_checkpoint_path = f"{experiment_dir}/{train_steps:07d}"
                os.makedirs(cloud_checkpoint_path, exist_ok=True)

                ### saving model parameters
                with FSDP.state_dict_type(
                    var_wo_ddp,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                ):
                    consolidated_model_state_dict = var_wo_ddp.state_dict()
                    if global_rank == 0:
                        consolidated_fn = "consolidated.pth"
                        torch.save(consolidated_model_state_dict, 
                        os.path.join(cloud_checkpoint_path, consolidated_fn))
                dist.barrier()
                del consolidated_model_state_dict
                logger.info(f"Saved consolidated to {cloud_checkpoint_path}")

                ### saving optimizer
                opt_state_fn = (
                    f"optimizer.{dist.get_rank():05d}-of-"
                    f"{dist.get_world_size():05d}.pth"
                )
                torch.save(optimizer.state_dict(), os.path.join(cloud_checkpoint_path, opt_state_fn))
                dist.barrier()
                logger.info(f"Saved optimizer to {cloud_checkpoint_path}")

                ### saving training step
                if global_rank == 0:
                    with open(os.path.join(cloud_checkpoint_path, "resume_step.txt"), "w") as f:
                        print(train_steps, file=f)
                dist.barrier()
                logger.info(f"Saved training step to {cloud_checkpoint_path}")



                cloud_checkpoint_path = f"{experiment_dir}/{train_steps:07d}"
                os.makedirs(cloud_checkpoint_path, exist_ok=True)


    ### saving final model parameters
    cloud_checkpoint_path = f"{experiment_dir}/final"
    os.makedirs(cloud_checkpoint_path, exist_ok=True)
    with FSDP.state_dict_type(
        var_wo_ddp,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        consolidated_model_state_dict = var_wo_ddp.state_dict()
        if global_rank == 0:
            consolidated_fn = "consolidated.pth"
            torch.save(consolidated_model_state_dict, 
            os.path.join(cloud_checkpoint_path, consolidated_fn))
    dist.barrier()
    del consolidated_model_state_dict
    logger.info(f"Saved consolidated to {cloud_checkpoint_path}")

    var_wo_ddp.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")

