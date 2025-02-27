


# d24
torchrun --nnodes=4 --nproc_per_node=8 --node_rank=$node_rank  --master_addr=$master_addr --master_port=$master_port  train.py \
                                                                 --depth=24 --bs=768 --ep=350 --tblr=8e-5 --fp16=1 --alng=1e-4 --wpe=0.01



# d20
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$node_rank  --master_addr=$master_addr --master_port=$master_port  train.py \
                                                                --depth=20 --bs=768 --ep=250 --fp16=1 --alng=1e-3 --wpe=0.1



# d16
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$node_rank  --master_addr=$master_addr --master_port=$master_port  train.py \
                                                                --depth=20 --bs=768 --ep=180 --fp16=1 --alng=1e-3 --wpe=0.1