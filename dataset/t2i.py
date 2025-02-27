import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image 
import time


class LaionasDataset(Dataset):
    def __init__(self, args, transform):
        img_path_list = []
        img_path_list = torch.load(os.path.join(args.data_path, "img_path_list.pt"))

        self.img_path_list = img_path_list
        self.transform = transform

        self.root = args.data_path
        self.saveroot = 'flant5-caption-en'

        self.image_size = args.image_size
        latent_size = args.image_size // args.downsample_size
        self.code_len = latent_size ** 2
        self.t5_feature_max_len = 120
        self.t5_feature_dim = 2048
        self.max_seq_length = self.t5_feature_max_len + self.code_len

    def __len__(self):
        return len(self.img_path_list)

    def dummy_data(self):
        img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
        attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).unsqueeze(0)
        valid = 0
        return img, t5_feat_padding, attn_mask, valid

    def __getitem__(self, index):
        dict_ = self.img_path_list[index]
        # img_path, code_dir, code_name = self.img_path_list[index]
        img_path = dict_['image_path']
        code_root = img_path.replace(self.root, '')
        t5_file = os.path.join(self.root, self.saveroot, code_root).replace('.jpg', '.npy')

        try:
            img = Image.open(img_path).convert("RGB")      
            t5_feat = torch.from_numpy(np.load(t5_file))
        except:
            print(f"'{img_path}' not exist...")
            trytime = 600
            start_time = time.time()
            while True:
                time.sleep(trytime)  
                elapsed_time = time.time() - start_time
                print(f"waiting for {elapsed_time / 60:.2f} min...")
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")      
                    t5_feat = torch.from_numpy(np.load(t5_file))
                    break

        img = self.transform(img)

        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
        # assert os.path.isfile(t5_file)
        # t5_feat = torch.randn(1, 23, 2048)

        t5_feat_len = t5_feat.shape[1] 
        feat_len = min(self.t5_feature_max_len, t5_feat_len)
        t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len]
        emb_mask = torch.zeros((self.t5_feature_max_len,))
        emb_mask[-feat_len:] = 1
        attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length))
        T = self.t5_feature_max_len
        attn_mask[:, :T] = attn_mask[:, :T] * emb_mask.unsqueeze(0)
        eye_matrix = torch.eye(self.max_seq_length, self.max_seq_length)
        attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
        attn_mask = attn_mask.unsqueeze(0).to(torch.bool)
        valid = 1

        # mask for mamba:
        attn_mask = torch.zeros(self.t5_feature_max_len)
        attn_mask[-t5_feat_len:] = 1.

        return img, t5_feat_padding, attn_mask, torch.tensor(valid)



def build_laionas(args, transform):
    return LaionasDataset(args, transform)

