import os
import random
import numpy as np

from sklearn.datasets import make_moons

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Sampler, Dataset, DataLoader


from parser import get_args
from ssl_lib.algs.builder import gen_ssl_alg
from ssl_lib.consistency.builder import gen_consistency
from ssl_lib.datasets.utils import InfiniteSampler

def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def gen_model(in_dims, out_dims, hidden_dims=256):
    return nn.Sequential(
        nn.Linear(in_dims, hidden_dims),
        nn.ReLU(),
        nn.Linear(hidden_dims, hidden_dims),
        nn.ReLU(),
        nn.Linear(hidden_dims, out_dims)
    )

def gen_ssl_moon_dataset(seed, num_samples, n_labeled, noise_factor=0.1):
    assert num_samples > n_labeled
    data, label = make_moons(num_samples, False, noise_factor, random_state=seed)
    data = (data - data.mean(0, keepdims=True)) / data.std(0, keepdims=True)

    l0_idx = (label == 0)
    l1_idx = (label == 1)

    l0_data = data[l0_idx]
    l1_data = data[l1_idx]

    l0_data = np.random.permutation(l0_data)
    l1_data = np.random.permutation(l1_data)

    l_data = np.concatenate([
        l0_data[:n_labeled//2],
        l1_data[:n_labeled//2]
    ])

    l_label = np.concatenate([
        np.zeros((n_labeled // 2,)), 
        np.ones((n_labeled // 2,))
    ])

    u_data    = np.concatenate([
        l0_data[n_labeled//2:], 
        l1_data[n_labeled//2:]
    ])

    u_label   = np.concatenate([
        np.ones((l0_data.shape[0] - (n_labeled//2),)),
        np.zeros((l1_data.shape[0] - (n_labeled//2),))
    ])

    return l_data, u_data, l_label, u_label


# -
# Cli

args = get_args()
set_seeds(args.seed)
if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"

# -
# Model(s)

# consistency type (really just CE or MSE, with soft and hard)
consistency = gen_consistency(args.consistency, args)

# semi-supervised algo
ssl_alg = gen_ssl_alg(args.alg, args)

# main (student) model
model     = gen_model(args.in_dims, args.num_classes, args.hidden_dims).to(device)

# teacher model
if args.ema_teacher:
    teacher_model = gen_model(args.in_dims, args.num_classes, args.hidden_dims).to(device)
    teacher_model.load_state_dict(model.state_dict())
else:
    teacher_model = None

# evaluation model
if args.weight_average:
    average_model = gen_model(args.in_dims, args.num_classes, args.hidden_dims).to(device)
    average_model.load_state_dict(model.state_dict())
else:
    average_model = None


# -
# Optimizer

optimizer    = optim.AdamW(model.parameters(), args.lr, (args.momentum, 0.999), weight_decay=0)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [400000, ], args.lr_decay_rate)

# -
# Data
l_data, u_data, l_label, u_label = gen_ssl_moon_dataset(
    args.seed, 1000, args.num_labels, 0.1
)

class TorchDataset(Dataset):
    """
    For labeled dataset
    """
    def __init__(self, data, labels, transform=None):
        self.data      = data
        self.labels    = labels
        self.transform = transform

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx]).float()
        label = int(self.labels[idx])
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.data)

l_dataset = TorchDataset(l_data, l_label)
u_dataset = TorchDataset(u_data, u_label)

l_loader = DataLoader(
    l_dataset,
    args.l_batch_size,
    sampler=InfiniteSampler(len(l_dataset), args.iteration * args.l_batch_size),
    num_workers=args.num_workers
)
u_loader = DataLoader(
    u_dataset,
    args.ul_batch_size,
    sampler=InfiniteSampler(len(u_dataset), args.iteration * args.ul_batch_size),
    num_workers=args.num_workers
)

u_loader_eval = DataLoader(
    u_dataset,
    args.ul_batch_size
)

w_aug = lambda x: x + torch.randn_like(x) * 0.1
s_aug = lambda x: x + torch.randn_like(x) * 0.2

# -
# Train
model.train()
for i, (l_data, u_data) in enumerate(zip(l_loader, u_loader)):
    print("start iter ", i)
    # get labeled / unlabeled weak aug / unlabeled strong aug
    l_data, l_labels  = l_data
    u_data, u_labels  = u_data

    l_data, l_labels = l_data.to(device), l_labels.to(device)
    u_data, u_labels = u_data.to(device), u_labels.to(device)

    u_w_data = w_aug(u_data)
    u_s_data = s_aug(u_data)

    all_data     = torch.cat([l_data, u_w_data, u_s_data], 0)
    base_forward = model.forward
    base_logits  = base_forward(all_data)
    
    # supervised loss
    l_preds      = base_logits[:l_labels.shape[0]]
    L_supervised = F.cross_entropy(l_preds, l_labels)

    if args.coef > 0:
        s_u_w_logits, s_u_s_logits = torch.chunk(base_logits[l_labels.shape[0]:], 2, dim=0)
        if teacher_model is not None:
            t_forward = teacher_model.forward
            t_logits  = teacher_forward(all_data)
            t_u_w_logits, _ = torch.chunk(t_logits[l_labels.shape[0]:], 2, dim=0)
        else:
            t_forward = base_forward
            t_u_w_logits = s_u_w_logits

        # consistency loss
        y, targets, mask = ssl_alg(
            stu_preds   = s_u_s_logits,
            tea_logits  = t_u_w_logits,
            data        = u_s_data,
            stu_forward = base_forward,
            tea_forward = t_forward
        )
        L_consistency = consistency(y, targets, mask, weak_prediction=t_u_w_logits.softmax(1))
    else:
        L_consistency = torch.zeros_like(L_supervised)
        mask = None

    # total loss
    loss = L_supervised + (args.coef*L_consistency)
    if args.entropy_minimization > 0:
        loss -= args.entropy_minimization * (s_u_w_logits.softmax(1) * F.softmax(s_u_w_logits, 1)).sum(1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if args.ema_teacher:
        model_utils.ema_update(
            teacher_model, model, args.ema_teacher_factor,
            args.weight_decay * cur_lr if args.ema_apply_wd else None, 
            cur_iteration if args.ema_teacher_warmup else None)
    if args.weight_average:
        model_utils.ema_update(
            average_model, model, args.wa_ema_factor, 
            args.weight_decay * cur_lr if args.wa_apply_wd else None)
    
    # eval
    if i % 2 == 0:

        model.eval()
        acc = 0
        for k, (u_data, u_labels) in enumerate(u_loader_eval):
            u_data, u_labels = u_data.to(device), u_labels.to(device)

            forward = model.forward
            logits  = forward(u_data)
            acc    += (logits.max(1)[1] == u_labels).float().mean()

            print(logits[0:5], u_labels[0:5])

        acc /= (k+1)
        print(f"acc after epoch {i}:", acc)




