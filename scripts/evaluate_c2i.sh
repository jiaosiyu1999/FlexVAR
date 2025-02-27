
# 06: 1_2_4_6_10_16
# 08: 1_2_3_5_7_10_13_16
# 09: 1_2_3_4_5_7_10_13_16
# 10: 1_2_3_4_5_6_8_10_13_16
# 11: 1_2_3_4_5_6_7_9_11_13_16
# 12: 1_2_3_4_5_6_7_8_10_12_14_16
# 13: 1_2_3_4_5_6_7_8_9_10_12_14_16
# 14: 1_2_3_4_5_6_7_8_9_10_11_12_14_16

# 32: 1_2_3_4_5_6_7_8_9_10_12_14_16_23_32


var_ckpt="out/VARd24/350epo/ar-ckpt-epo349.pth"
args_infer_patch_nums="1_2_3_4_5_7_10_13_16" # 1_2_3_4_5_6_8_10_13_16
args_cfg=2.5
args_maxpn=16
depth=24


export TOKENIZERS_PARALLELISM=false 
torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0  eval_c2i.py  --batch_size 16  --cfg $args_cfg --top_k 900 \
                                                                   --maxpn $args_maxpn  --infer_patch_nums $args_infer_patch_nums \
                                                                   --var_ckpt $var_ckpt --depth $depth


# 提取 epo 值
epo=$(basename "$var_ckpt" | cut -d'.' -f1 | sed 's/.*epo//')
# 处理 infer_patch_nums
infer_patch_nums=$(echo "$args_infer_patch_nums" | tr '-' '_')
IFS='_' read -ra infer_patch_nums_array <<< "$infer_patch_nums"
# 计算 shape 和 step 数量
shape=$(( ${infer_patch_nums_array[-1]} * 16 ))
steps=${#infer_patch_nums_array[@]}
# 构建 save_root 路径
save_root="vis/eval_c2i/d${depth}_cfg${args_cfg}_shape${shape}_${steps}step_maxpn${args_maxpn}.npz"
echo "$save_root"
echo "save_root: $save_root"


python  utils/evaluations/c2i/evaluator.py --sample_batch  $save_root

# CUDA_VISIBLE_DEVICES=1  python  utils/evaluations/c2i/evaluator.py --sample_batch vis/eval_c2i/d24_epo349_cfg2.5_shape256_8step_maxpn16