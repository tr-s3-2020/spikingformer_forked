import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial
import numpy as np
import copy
import pdb

__all__ = ['Spikingformer']

save = None
power = 2.84 # mW
frequency = 50 #MHz
batch_id = 0
total_compute_latency = 0
total_compute_latency_delta = 0
ich_parallel_num = 8
ema = torch.zeros(4,1)
total_spike_all = 0
total_OPS = 0
timewindow = 3

def calculate_latency_for_qkvmm_nonlif_bl(a, b, add_in_width0, add_in_width1): # matrix multiplication latency baseline, calculate a * b. add_in_width is the addition input width of adder tree.
    T, B, head, H, W = a.shape
    # maximum H and W based on PE and on-chip buffer size
    buffer_size = 32 * 1024
    Max_H = 256
    Max_W = buffer_size / (16 * add_in_width1 * T / 8 + 256 * add_in_width0 * T / 8)
    load_latency = 0
    compute_latency_pT = 0
    compute_latency = 0
    store_latency_pT = 0
    store_latency = 0
    if H <= Max_H or W <= Max_W:
        load_latency = (H * W * add_in_width0 * T + H * W * add_in_width1 * T) / 32 # overall bits / bandwidth
        compute_latency_pT = (1 + H + 4) * np.ceil(W/16)
        compute_latency = compute_latency_pT * T
        store_latency_pT = (H * H * 16) / 32
        store_latency = store_latency_pT * T
        latency_a = load_latency + compute_latency + store_latency
        # print(latency_a)
        latency_a = latency_a * head
    latency_a = latency_a * B
    return latency_a

def calculate_latency_for_qkvmm_nonlif_sp(a, b, add_in_width0, add_in_width1): # matrix multiplication latency baseline, calculate a * b. add_in_width is the addition input width of adder tree.
    T, B, head, H, W = a.shape
    # maximum H and W based on PE and on-chip buffer size
    buffer_size = 32 * 1024
    Max_H = 256
    Max_W = buffer_size / (16 * add_in_width1 * T / 8 + 256 * add_in_width0 * T / 8)
    load_latency = 0
    compute_latency_pT = 0
    compute_latency = 0
    store_latency_pT = 0
    store_latency = 0
    if H <= Max_H or W <= Max_W:
        load_latency = (H * W * add_in_width0 * T + H * W * add_in_width1 * T) / 32 # overall bits / bandwidth
        compute_latency_pT = (1 + H + 4) * np.ceil(W/16)
        compute_latency = compute_latency_pT * T
        store_latency_pT = (H * H * 16) / 32
        store_latency = store_latency_pT * T
        latency_a = load_latency + compute_latency + store_latency
        # print(latency_a)
        latency_a = latency_a * head
    latency_a = latency_a * B
    return latency_a

def calculate_latency_for_qkvmm_lif_bl(a, b, add_in_width0, add_in_width1): # matrix multiplication latency baseline, calculate a * b. add_in_width is the addition input width of adder tree.
    T, B, head, H, W = a.shape
    # maximum H and W based on PE and on-chip buffer size
    buffer_size = 32 * 1024
    Max_H = 256
    Max_W = buffer_size / (16 * add_in_width1 * T / 8 + 256 * add_in_width0 * T / 8)
    load_latency = 0
    compute_latency_pT = 0
    compute_latency = 0
    lif_latency_pT = 0
    lif_latency = 0
    store_latency_pT = 0
    store_latency = 0
    if H <= Max_H or W <= Max_W:
        load_latency = (H * W * add_in_width0 * T + H * W * add_in_width1 * T) / 32 # overall bits / bandwidth
        compute_latency_pT = (1 + H + 4) * np.ceil(W/16)
        compute_latency = compute_latency_pT * T
        lif_latency_pT= H
        lif_latency = lif_latency_pT * T
        store_latency_pT = (H * H) / 32
        store_latency = store_latency_pT * T
        latency_a = load_latency + compute_latency + lif_latency + store_latency
        # print(latency_a)
        latency_a = latency_a * head
    latency_a = latency_a * B
    return latency_a

def calculate_latency_for_qkvgen_lif_bl(a, b, add_in_width0, add_in_width1): # matrix multiplication latency baseline, calculate a * b. add_in_width is the addition input width of adder tree.
    T1, B1, H1, W1 = a.shape
    T2, B2, H2, W2 = b.shape
    W2 = 3 * W2
    # maximum H and W based on PE and on-chip buffer size
    buffer_size = 32 * 1024
    Max_H1 = 256
    Max_W2 = 256
    Max_W1_H2 = buffer_size / (256 * add_in_width0 / 8 + 16 * add_in_width1 / 8)
    load_latency = 0
    compute_latency_pT = 0
    compute_latency = 0
    lif_latency_pT = 0
    lif_latency = 0
    store_latency_pT = 0
    store_latency = 0
    if (H1 <= Max_H1 and W1 <= Max_W1_H2) or (H2 <= Max_W1_H2 and W2 <= Max_W2):
        load_latency = (H1 * W1 * add_in_width0 * T1 + H2 * W2 * add_in_width1) / 32 # overall bits / bandwidth
        compute_latency_pT = (1 + H1 + 4) * np.ceil(W1/16) * np.ceil(W2/256)
        compute_latency = compute_latency_pT * T1
        lif_latency_pT= H1 * np.ceil(W2/256)
        lif_latency = lif_latency_pT * T1
        store_latency_pT = (H1 * W2) / 32
        store_latency = store_latency_pT * T1
        latency_a = load_latency + compute_latency + lif_latency + store_latency
        # print(latency_a)
    elif (H1 <= Max_H1) or (W2 <= Max_W2): # activation first
        load_latency = (H1 * W1 * add_in_width0 * T1 * np.ceil(W2/256) + H2 * W2 * add_in_width1) / 32 # overall bits / bandwidth
        compute_latency_pT = (1 + H1 + 4) * np.ceil(W1/16) * np.ceil(W2/256)
        compute_latency = compute_latency_pT * T1
        lif_latency_pT= H1 * np.ceil(W2/256)
        lif_latency = lif_latency_pT * T1
        store_latency_pT = (H1 * W2) / 32
        store_latency = store_latency_pT * T1
        latency_a = load_latency + compute_latency + lif_latency + store_latency
    latency_a = latency_a * B1
    return latency_a

def calculate_latency_for_proj_lif_bl(a, b, add_in_width0, add_in_width1): # matrix multiplication latency baseline, calculate a * b. add_in_width is the addition input width of adder tree.
    T1, B1, H1, W1 = a.shape
    H2, W2 = b.shape
    # maximum H and W based on PE and on-chip buffer size
    buffer_size = 32 * 1024
    Max_H1 = 256
    Max_W2 = 256
    Max_W1_H2 = buffer_size / (256 * add_in_width0 / 8 + 16 * add_in_width1 / 8)
    load_latency = 0
    compute_latency_pT = 0
    compute_latency = 0
    lif_latency_pT = 0
    lif_latency = 0
    store_latency_pT = 0
    store_latency = 0
    if (H1 <= Max_H1 and W1 <= Max_W1_H2) or (H2 <= Max_W1_H2 and W2 <= Max_W2):
        load_latency = (H1 * W1 * add_in_width0 * T1 + H2 * W2 * add_in_width1) / 32 # overall bits / bandwidth
        compute_latency_pT = (1 + H1 + 4) * np.ceil(W1/16) * np.ceil(W2/256)
        compute_latency = compute_latency_pT * T1
        lif_latency_pT= H1 * np.ceil(W2/256)
        lif_latency = lif_latency_pT * T1
        store_latency_pT = (H1 * W2) / 32
        store_latency = store_latency_pT * T1
        latency_a = load_latency + compute_latency + lif_latency + store_latency
        # print(latency_a)
    elif (H1 <= Max_H1) or (W2 <= Max_W2): # activation first
        load_latency = (H1 * W1 * add_in_width0 * T1 * np.ceil(W2/256) + H2 * W2 * add_in_width1) / 32 # overall bits / bandwidth
        compute_latency_pT = (1 + H1 + 4) * np.ceil(W1/16) * np.ceil(W2/256)
        compute_latency = compute_latency_pT * T1
        lif_latency_pT= H1 * np.ceil(W2/256)
        lif_latency = lif_latency_pT * T1
        store_latency_pT = (H1 * W2) / 32
        store_latency = store_latency_pT * T1
        latency_a = load_latency + compute_latency + lif_latency + store_latency
    latency_a = latency_a * B1
    load_latency = load_latency * B1
    compute_latency = compute_latency * B1
    lif_latency = lif_latency * B1
    store_latency = store_latency * B1
    return latency_a, load_latency, compute_latency, lif_latency, store_latency

def calculate_latency_for_proj_lif_delta_i(a, b, add_in_width1, SCF): # input, load and compute, SCF: sparse compress format
    bank_num = 16
    T, H1, W1 = a.shape # batch size should be 1
    H2, W2 = b.shape
    # maximum H and W based on PE and on-chip buffer size
    buffer_size = 32 * 1024
    Max_H1 = 256
    Max_W2 = 256
    # calculate the overall memory consumption
    # find the maximum length of nonzero value in each timestep in a_Hreshape
    all_nonzero_idx = a.nonzero()
    # group by timestep
    all_nonzero_idx_t = [[] for i in range(T)]
    for each in all_nonzero_idx.tolist():
        all_nonzero_idx_t[each[0]].append([each[1], each[2]])
    max_len_a_Hreshape_t = max([len(each) for each in all_nonzero_idx_t])
    act_volume_for_mem_cons = 0
    # sparse encoding: depend on SCF
    if SCF == 'COO':
        bw = 16
        coo_x_bw = bw
        coo_y_bw = bw
        act_volume_for_mem_cons = (coo_x_bw + coo_y_bw) * max_len_a_Hreshape_t
    elif SCF == 'CSR':
        bw = 16
        csr_row_ptr_vol = (H1 + 1) * bw
        csr_col_idx_vol = max_len_a_Hreshape_t * bw
        act_volume_for_mem_cons = csr_row_ptr_vol + csr_col_idx_vol
    elif SCF == 'AdaptiveCSR':
        acsr_row_ptr_vol = (H1 + 1) * np.ceil(np.log2(max_len_a_Hreshape_t))
        acsr_col_idx_vol = np.ceil(np.log2(W1)) * max_len_a_Hreshape_t
        act_volume_for_mem_cons = acsr_row_ptr_vol + acsr_col_idx_vol
    mem_cons_full_T = (act_volume_for_mem_cons + add_in_width1 * len(torch.unique(all_nonzero_idx[:,2])) * 16) / 8
    # print(mem_cons_full_T)
    load_latency = 0
    compute_latency_pT = 0
    compute_latency = 0
    lif_latency_pT = 0
    lif_latency = 0
    latency_per_token = torch.zeros(H1)
    if (H1 <= Max_H1) or (W2 <= Max_W2):
        if mem_cons_full_T <= buffer_size:
            # calculate latency
            for t in range(T):
                act_volume = 0
                # sparse encoding: depend on SCF
                if SCF == 'COO':
                    bw = 16
                    coo_x_bw = bw
                    coo_y_bw = bw
                    act_volume = (coo_x_bw + coo_y_bw) * len(all_nonzero_idx_t[t])
                elif SCF == 'CSR':
                    bw = 16
                    csr_row_ptr_vol = (H1 + 1) * bw
                    csr_col_idx_vol = len(all_nonzero_idx_t[t]) * bw
                    act_volume = csr_row_ptr_vol + csr_col_idx_vol
                elif SCF == 'AdaptiveCSR':
                    acsr_row_ptr_vol = (H1 + 1) * np.ceil(np.log2(len(all_nonzero_idx_t[t])))
                    acsr_col_idx_vol = np.ceil(np.log2(W1)) * len(all_nonzero_idx_t[t])
                    act_volume = acsr_row_ptr_vol + acsr_col_idx_vol
                load_latency += (act_volume + add_in_width1 * len(torch.unique(all_nonzero_idx[:,2])) * 16) / 8 / 32 # overall bits / bandwidth
                for h in range(H1):
                    if (a[t, h, :] != 0).sum() != 0: # if the h-th token has non-zero value
                        nonzero_ich_idx = a[t, h, :].nonzero()[:,0]
                        weight_raddr = (nonzero_ich_idx % bank_num).int()
                        latency_per_token[h] = (1 + 1 + 4) * weight_raddr.unique(return_counts=True)[1].max().item()
                compute_latency_pT = latency_per_token.sum() * np.ceil(W2/256)
                compute_latency += compute_latency_pT
            lif_latency_pT= H1 * np.ceil(W2/256)
            lif_latency = lif_latency_pT * T
            latency_l_c = load_latency + compute_latency + lif_latency
        else: # activation first
            # calculate latency
            for t in range(T):
                act_volume = 0
                # sparse encoding: depend on SCF
                if SCF == 'COO':
                    bw = 16
                    coo_x_bw = bw
                    coo_y_bw = bw
                    act_volume = (coo_x_bw + coo_y_bw) * len(all_nonzero_idx_t[t])
                elif SCF == 'CSR':
                    bw = 16
                    csr_row_ptr_vol = (H1 + 1) * bw
                    csr_col_idx_vol = len(all_nonzero_idx_t[t]) * bw
                    act_volume = csr_row_ptr_vol + csr_col_idx_vol
                elif SCF == 'AdaptiveCSR':
                    acsr_row_ptr_vol = (H1 + 1) * np.ceil(np.log2(len(all_nonzero_idx_t[t])))
                    acsr_col_idx_vol = np.ceil(np.log2(W1)) * len(all_nonzero_idx_t[t])
                    act_volume = acsr_row_ptr_vol + acsr_col_idx_vol
                load_latency += (act_volume * np.ceil(mem_cons_full_T / buffer_size) + add_in_width1 * len(torch.unique(all_nonzero_idx[:,1])) * 16) / 8 / 32 # overall bits / bandwidth
                for h in range(H1):
                    if (a[t, h, :] != 0).sum() != 0: # if the h-th token has non-zero value
                        nonzero_ich_idx = a[t, h, :].nonzero()[:,0]
                        weight_raddr = (nonzero_ich_idx % bank_num).int()
                        latency_per_token[h] = (1 + 1 + 4) * weight_raddr.unique(return_counts=True)[1].max().item()
                compute_latency_pT = latency_per_token.sum() * np.ceil(W2/256)
                compute_latency += compute_latency_pT
            lif_latency_pT= H1 * np.ceil(W2/256)
            lif_latency = lif_latency_pT * T
            latency_l_c = load_latency + compute_latency + lif_latency
    return latency_l_c, load_latency, compute_latency, lif_latency

def calculate_latency_for_proj_lif_delta_o(a, SCF):
    T, H1, W1 = a.shape # batch size should be 1
    all_nonzero_idx = a.nonzero()
    act_volume = 0
    # sparse encoding: depend on SCF
    if SCF == 'COO':
        bw = 16
        coo_x_bw = bw
        coo_y_bw = bw
        act_volume = (coo_x_bw + coo_y_bw) * len(all_nonzero_idx)
    elif SCF == 'CSR':
        bw = 16
        csr_row_ptr_vol = (H1 + 1) * bw
        csr_col_idx_vol = len(all_nonzero_idx) * bw
        act_volume = csr_row_ptr_vol + csr_col_idx_vol
    elif SCF == 'AdaptiveCSR':
        acsr_row_ptr_vol = (H1 + 1) * np.ceil(np.log2(len(all_nonzero_idx)))
        acsr_col_idx_vol = np.ceil(np.log2(W1)) * len(all_nonzero_idx)
        act_volume = acsr_row_ptr_vol + acsr_col_idx_vol
    latency_s = (act_volume) / 8 / 32
    return latency_s

def calculate_latency_for_mlp_lif_bl(a, b, add_in_width0, add_in_width1): # matrix multiplication latency baseline, calculate a * b. add_in_width is the addition input width of adder tree.
    T1, B, W1, Hh1, Hw1 = a.shape
    W2, H2 = b.shape
    H1 = Hh1 * Hw1
    # maximum H and W based on PE and on-chip buffer size
    buffer_size = 32 * 1024
    Max_H1 = 256
    Max_W2 = 256
    Max_W1_H2 = buffer_size / (256 * add_in_width0 / 8 + 16 * add_in_width1 / 8)
    load_latency = 0
    compute_latency_pT = 0
    compute_latency = 0
    lif_latency_pT = 0
    lif_latency = 0
    store_latency_pT = 0
    store_latency = 0
    if (H1 <= Max_H1 and W1 <= Max_W1_H2) or (H2 <= Max_W1_H2 and W2 <= Max_W2):
        load_latency = (H1 * W1 * add_in_width0 * T1 + H2 * W2 * add_in_width1) / 32 # overall bits / bandwidth
        compute_latency_pT = (1 + H1 + 4) * np.ceil(W1/16) * np.ceil(W2/256)
        compute_latency = compute_latency_pT * T1
        lif_latency_pT= H1 * np.ceil(W2/256)
        lif_latency = lif_latency_pT * T1
        store_latency_pT = (H1 * W2) / 32
        store_latency = store_latency_pT * T1
        latency_a = load_latency + compute_latency + lif_latency + store_latency
        # print(latency_a)
    elif (H1 <= Max_H1) or (W2 <= Max_W2): # activation first
        load_latency = (H1 * W1 * add_in_width0 * T1 * np.ceil(W2/256) + H2 * W2 * add_in_width1) / 32 # overall bits / bandwidth
        compute_latency_pT = (1 + H1 + 4) * np.ceil(W1/16) * np.ceil(W2/256)
        compute_latency = compute_latency_pT * T1
        lif_latency_pT= H1 * np.ceil(W2/256)
        lif_latency = lif_latency_pT * T1
        store_latency_pT = (H1 * W2) / 32
        store_latency = store_latency_pT * T1
        latency_a = load_latency + compute_latency + lif_latency + store_latency
    latency_a = latency_a * B
    load_latency = load_latency * B
    compute_latency = compute_latency * B
    lif_latency = lif_latency * B
    store_latency = store_latency * B
    return latency_a, load_latency, compute_latency, lif_latency, store_latency

def calculate_latency_for_mlp_lif_delta_i(a, b, add_in_width1, SCF): # input, load and compute, SCF: sparse compress format
    bank_num = 16
    T, W1, Hh1, Hw1 = a.shape # batch size should be 1
    W2, H2 = b.shape
    a_Hreshape = a.reshape(T, W1, -1)
    T, W1, H1 = a_Hreshape.shape
    # maximum H and W based on PE and on-chip buffer size
    buffer_size = 32 * 1024
    Max_H1 = 256
    Max_W2 = 256
    # calculate the overall memory consumption
    # find the maximum length of nonzero value in each timestep in a_Hreshape
    all_nonzero_idx = a_Hreshape.nonzero()
    # group by timestep
    all_nonzero_idx_t = [[] for i in range(T)]
    for each in all_nonzero_idx.tolist():
        all_nonzero_idx_t[each[0]].append([each[1], each[2]])
    max_len_a_Hreshape_t = max([len(each) for each in all_nonzero_idx_t])
    act_volume_for_mem_cons = 0
    # sparse encoding: depend on SCF
    if SCF == 'COO':
        bw = 16
        coo_x_bw = bw
        coo_y_bw = bw
        act_volume_for_mem_cons = (coo_x_bw + coo_y_bw) * max_len_a_Hreshape_t
    elif SCF == 'CSR':
        bw = 16
        csr_row_ptr_vol = (H1 + 1) * bw
        csr_col_idx_vol = max_len_a_Hreshape_t * bw
        act_volume_for_mem_cons = csr_row_ptr_vol + csr_col_idx_vol
    elif SCF == 'AdaptiveCSR':
        acsr_row_ptr_vol = (H1 + 1) * np.ceil(np.log2(max_len_a_Hreshape_t))
        acsr_col_idx_vol = np.ceil(np.log2(W1)) * max_len_a_Hreshape_t
        act_volume_for_mem_cons = acsr_row_ptr_vol + acsr_col_idx_vol
    mem_cons_full_T = (act_volume_for_mem_cons + add_in_width1 * len(torch.unique(all_nonzero_idx[:,1])) * 16) / 8
    # print(mem_cons_full_T)
    load_latency = 0
    compute_latency_pT = 0
    compute_latency = 0
    lif_latency_pT = 0
    lif_latency = 0
    latency_per_token = torch.zeros(H1)
    if (H1 <= Max_H1) or (W2 <= Max_W2):
        if mem_cons_full_T <= buffer_size:
            # calculate latency
            for t in range(T):
                act_volume = 0
                # sparse encoding: depend on SCF
                if SCF == 'COO':
                    bw = 16
                    coo_x_bw = bw
                    coo_y_bw = bw
                    act_volume = (coo_x_bw + coo_y_bw) * len(all_nonzero_idx_t[t])
                elif SCF == 'CSR':
                    bw = 16
                    csr_row_ptr_vol = (H1 + 1) * bw
                    csr_col_idx_vol = len(all_nonzero_idx_t[t]) * bw
                    act_volume = csr_row_ptr_vol + csr_col_idx_vol
                elif SCF == 'AdaptiveCSR':
                    acsr_row_ptr_vol = (H1 + 1) * np.ceil(np.log2(len(all_nonzero_idx_t[t])))
                    acsr_col_idx_vol = np.ceil(np.log2(W1)) * len(all_nonzero_idx_t[t])
                    act_volume = acsr_row_ptr_vol + acsr_col_idx_vol
                load_latency += (act_volume + add_in_width1 * len(torch.unique(all_nonzero_idx[:,1])) * 16) / 8 / 32 # overall bits / bandwidth
                for h in range(H1):
                    if (a_Hreshape[t, :, h] != 0).sum() != 0: # if the h-th token has non-zero value
                        nonzero_ich_idx = a_Hreshape[t, :, h].nonzero()[:,0]
                        weight_raddr = (nonzero_ich_idx % bank_num).int()
                        latency_per_token[h] = (1 + 1 + 4) * weight_raddr.unique(return_counts=True)[1].max().item()
                compute_latency_pT = latency_per_token.sum() * np.ceil(W2/256)
                compute_latency += compute_latency_pT
            lif_latency_pT= H1 * np.ceil(W2/256)
            lif_latency = lif_latency_pT * T
            latency_l_c = load_latency + compute_latency + lif_latency
        else: # activation first
            # calculate latency
            for t in range(T):
                act_volume = 0
                # sparse encoding: depend on SCF
                if SCF == 'COO':
                    bw = 16
                    coo_x_bw = bw
                    coo_y_bw = bw
                    act_volume = (coo_x_bw + coo_y_bw) * len(all_nonzero_idx_t[t])
                elif SCF == 'CSR':
                    bw = 16
                    csr_row_ptr_vol = (H1 + 1) * bw
                    csr_col_idx_vol = len(all_nonzero_idx_t[t]) * bw
                    act_volume = csr_row_ptr_vol + csr_col_idx_vol
                elif SCF == 'AdaptiveCSR':
                    acsr_row_ptr_vol = (H1 + 1) * np.ceil(np.log2(len(all_nonzero_idx_t[t])))
                    acsr_col_idx_vol = np.ceil(np.log2(W1)) * len(all_nonzero_idx_t[t])
                    act_volume = acsr_row_ptr_vol + acsr_col_idx_vol
                load_latency += (act_volume * np.ceil(mem_cons_full_T / buffer_size) + add_in_width1 * len(torch.unique(all_nonzero_idx[:,1])) * 16) / 8 / 32 # overall bits / bandwidth
                for h in range(H1):
                    if (a_Hreshape[t, :, h] != 0).sum() != 0: # if the h-th token has non-zero value
                        nonzero_ich_idx = a_Hreshape[t, :, h].nonzero()[:,0]
                        weight_raddr = (nonzero_ich_idx % bank_num).int()
                        latency_per_token[h] = (1 + 1 + 4) * weight_raddr.unique(return_counts=True)[1].max().item()
                compute_latency_pT = latency_per_token.sum() * np.ceil(W2/256)
                compute_latency += compute_latency_pT
            lif_latency_pT= H1 * np.ceil(W2/256)
            lif_latency = lif_latency_pT * T
            latency_l_c = load_latency + compute_latency + lif_latency
    return latency_l_c, load_latency, compute_latency, lif_latency

def calculate_latency_for_mlp_lif_delta_o(a, SCF):
    T, W1, Hh1, Hw1 = a.shape # batch size should be 1
    a_Hreshape = a.reshape(T, W1, -1)
    T, W1, H1 = a_Hreshape.shape
    all_nonzero_idx = a_Hreshape.nonzero()
    act_volume = 0
    # sparse encoding: depend on SCF
    if SCF == 'COO':
        bw = 16
        coo_x_bw = bw
        coo_y_bw = bw
        act_volume = (coo_x_bw + coo_y_bw) * len(all_nonzero_idx)
    elif SCF == 'CSR':
        bw = 16
        csr_row_ptr_vol = (H1 + 1) * bw
        csr_col_idx_vol = len(all_nonzero_idx) * bw
        act_volume = csr_row_ptr_vol + csr_col_idx_vol
    elif SCF == 'AdaptiveCSR':
        acsr_row_ptr_vol = (H1 + 1) * np.ceil(np.log2(len(all_nonzero_idx)))
        acsr_col_idx_vol = np.ceil(np.log2(W1)) * len(all_nonzero_idx)
        act_volume = acsr_row_ptr_vol + acsr_col_idx_vol
    latency_s = (act_volume) / 8 / 32
    return latency_s

# def calculate_latency_for_mlp_bl(a, b, add_in_width0, add_in_width1): # matrix multiplication latency baseline, calculate a * b. add_in_width is the addition input width of adder tree.
#     T1, B, W1, H1 = a.shape
#     W2, H2 = b.shape
#     # maximum H and W based on PE and on-chip buffer size
#     buffer_size = 32 * 1024
#     Max_H1 = 256
#     Max_W2 = 256
#     Max_W1_H2 = buffer_size / (256 * add_in_width0 / 8 + 16 * add_in_width1 / 8)
#     load_latency = 0
#     compute_latency_pT = 0
#     compute_latency = 0
#     lif_latency_pT = 0
#     lif_latency = 0
#     store_latency_pT = 0
#     store_latency = 0
#     if (H1 <= Max_H1 and W1 <= Max_W1_H2) or (H2 <= Max_W1_H2 and W2 <= Max_W2):
#         load_latency = (H1 * W1 * add_in_width0 * T1 + H2 * W2 * add_in_width1) / 32 # overall bits / bandwidth
#         compute_latency_pT = (1 + H1 + 4) * np.ceil(W1/16) * np.ceil(W2/256)
#         compute_latency = compute_latency_pT * T1
#         store_latency_pT = (H1 * W2 * 16) / 32
#         store_latency = store_latency_pT * T1
#         latency_a = load_latency + compute_latency + lif_latency + store_latency
#         # print(latency_a)
#     elif (H1 <= Max_H1) or (W2 <= Max_W2): # activation first
#         load_latency = (H1 * W1 * add_in_width0 * T1 * np.ceil(W2/256) + H2 * W2 * add_in_width1) / 32 # overall bits / bandwidth
#         compute_latency_pT = (1 + H1 + 4) * np.ceil(W1/16) * np.ceil(W2/256)
#         compute_latency = compute_latency_pT * T1
#         store_latency_pT = (H1 * W2 * 16) / 32
#         store_latency = store_latency_pT * T1
#         latency_a = load_latency + compute_latency + lif_latency + store_latency
#     latency_a = latency_a * B
#     return latency_a

# def calculate_latency_for_mlp_delta_i(a, b, add_in_width1, SCF): # input, load and compute
#     bank_num = 16
#     T, W1, Hh1, Hw1 = a.shape # batch size should be 1
#     W2, H2 = b.shape
#     a_Hreshape = a.reshape(T, W1, -1)
#     T, W1, H1 = a_Hreshape.shape
#     # maximum H and W based on PE and on-chip buffer size
#     buffer_size = 32 * 1024
#     Max_H1 = 256
#     Max_W2 = 256
#     # calculate the overall memory consumption
#     # find the maximum length of nonzero value in each timestep in a_Hreshape
#     all_nonzero_idx = a_Hreshape.nonzero()
#     # group by timestep
#     all_nonzero_idx_t = [[] for i in range(T)]
#     for each in all_nonzero_idx.tolist():
#         all_nonzero_idx_t[each[0]].append([each[1], each[2]])
#     max_len_a_Hreshape_t = max([len(each) for each in all_nonzero_idx_t])
#     act_volume_for_mem_cons = 0
#     # sparse encoding: depend on SCF
#     if SCF == 'COO':
#         bw = 16
#         coo_x_bw = bw
#         coo_y_bw = bw
#         act_volume_for_mem_cons = (coo_x_bw + coo_y_bw) * max_len_a_Hreshape_t
#     elif SCF == 'CSR':
#         bw = 16
#         csr_row_ptr_vol = (H1 + 1) * bw
#         csr_col_idx_vol = max_len_a_Hreshape_t * bw
#         act_volume_for_mem_cons = csr_row_ptr_vol + csr_col_idx_vol
#     elif SCF == 'AdaptiveCSR':
#         acsr_row_ptr_vol = (H1 + 1) * np.ceil(np.log2(max_len_a_Hreshape_t))
#         acsr_col_idx_vol = np.ceil(np.log2(W1)) * max_len_a_Hreshape_t
#         act_volume_for_mem_cons = acsr_row_ptr_vol + acsr_col_idx_vol
#     mem_cons_full_T = (act_volume_for_mem_cons + add_in_width1 * len(torch.unique(all_nonzero_idx[:,1])) * 16) / 8
#     # print(mem_cons_full_T)
#     load_latency = 0
#     compute_latency_pT = 0
#     compute_latency = 0
#     latency_per_token = torch.zeros(H1)
#     if (H1 <= Max_H1) or (W2 <= Max_W2):
#         if mem_cons_full_T <= buffer_size:
#             # calculate latency
#             for t in range(T):
#                 act_volume = 0
#                 # sparse encoding: depend on SCF
#                 if SCF == 'COO':
#                     bw = 16
#                     coo_x_bw = bw
#                     coo_y_bw = bw
#                     act_volume = (coo_x_bw + coo_y_bw) * len(all_nonzero_idx_t[t])
#                 elif SCF == 'CSR':
#                     bw = 16
#                     csr_row_ptr_vol = (H1 + 1) * bw
#                     csr_col_idx_vol = len(all_nonzero_idx_t[t]) * bw
#                     act_volume = csr_row_ptr_vol + csr_col_idx_vol
#                 elif SCF == 'AdaptiveCSR':
#                     acsr_row_ptr_vol = (H1 + 1) * np.ceil(np.log2(len(all_nonzero_idx_t[t])))
#                     acsr_col_idx_vol = np.ceil(np.log2(W1)) * len(all_nonzero_idx_t[t])
#                     act_volume = acsr_row_ptr_vol + acsr_col_idx_vol
#                 load_latency += (act_volume + add_in_width1 * len(torch.unique(all_nonzero_idx[:,1])) * 16) / 8 / 32 # overall bits / bandwidth
#                 for h in range(H1):
#                     if (a_Hreshape[t, :, h] != 0).sum() != 0: # if the h-th token has non-zero value
#                         nonzero_ich_idx = a_Hreshape[t, :, h].nonzero()[:,0]
#                         weight_raddr = (nonzero_ich_idx % bank_num).int()
#                         latency_per_token[h] = (1 + 1 + 4) * weight_raddr.unique(return_counts=True)[1].max().item()
#                 compute_latency_pT = latency_per_token.sum() * np.ceil(W2/256)
#                 compute_latency += compute_latency_pT
#             latency_l_c = load_latency + compute_latency
#         else: # activation first
#             # calculate latency
#             for t in range(T):
#                 act_volume = 0
#                 # sparse encoding: depend on SCF
#                 if SCF == 'COO':
#                     bw = 16
#                     coo_x_bw = bw
#                     coo_y_bw = bw
#                     act_volume = (coo_x_bw + coo_y_bw) * len(all_nonzero_idx_t[t])
#                 elif SCF == 'CSR':
#                     bw = 16
#                     csr_row_ptr_vol = (H1 + 1) * bw
#                     csr_col_idx_vol = len(all_nonzero_idx_t[t]) * bw
#                     act_volume = csr_row_ptr_vol + csr_col_idx_vol
#                 elif SCF == 'AdaptiveCSR':
#                     acsr_row_ptr_vol = (H1 + 1) * np.ceil(np.log2(len(all_nonzero_idx_t[t])))
#                     acsr_col_idx_vol = np.ceil(np.log2(W1)) * len(all_nonzero_idx_t[t])
#                     act_volume = acsr_row_ptr_vol + acsr_col_idx_vol
#                 load_latency += (act_volume * np.ceil(mem_cons_full_T / buffer_size) + add_in_width1 * len(torch.unique(all_nonzero_idx[:,1])) * 16) / 8 / 32 # overall bits / bandwidth
#                 for h in range(H1):
#                     if (a_Hreshape[t, :, h] != 0).sum() != 0: # if the h-th token has non-zero value
#                         nonzero_ich_idx = a_Hreshape[t, :, h].nonzero()[:,0]
#                         weight_raddr = (nonzero_ich_idx % bank_num).int()
#                         latency_per_token[h] = (1 + 1 + 4) * weight_raddr.unique(return_counts=True)[1].max().item()
#                 compute_latency_pT = latency_per_token.sum() * np.ceil(W2/256)
#                 compute_latency += compute_latency_pT
#             latency_l_c = load_latency + compute_latency
#     return latency_l_c

# def calculate_latency_for_mlp_delta_o(a, SCF):
    T, W1, Hh1, Hw1 = a.shape # batch size should be 1
    a_Hreshape = a.reshape(T, W1, -1)
    T, W1, H1 = a_Hreshape.shape
    all_nonzero_idx = a_Hreshape.nonzero()
    act_volume = 0
    # sparse encoding: depend on SCF
    if SCF == 'COO':
        bw = 16
        coo_x_bw = bw
        coo_y_bw = bw
        act_volume = (coo_x_bw + coo_y_bw) * len(all_nonzero_idx)
    elif SCF == 'CSR':
        bw = 16
        csr_row_ptr_vol = (H1 + 1) * bw
        csr_col_idx_vol = len(all_nonzero_idx) * bw
        act_volume = csr_row_ptr_vol + csr_col_idx_vol
    elif SCF == 'AdaptiveCSR':
        acsr_row_ptr_vol = (H1 + 1) * np.ceil(np.log2(len(all_nonzero_idx)))
        acsr_col_idx_vol = np.ceil(np.log2(W1)) * len(all_nonzero_idx)
        act_volume = acsr_row_ptr_vol + acsr_col_idx_vol
    latency_s = act_volume / 8 / 32
    return latency_s
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)

        self.mlp2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        latency_a_bl = 0
        latency_load_bl = 0
        latency_compute_bl = 0
        latency_lif_bl = 0
        latency_store_bl = 0
        
        latency_a_delta = 0
        latency_load_delta = 0
        latency_compute_delta = 0
        latency_lif_delta = 0
        latency_store_delta = 0
        
        latency_a_delta_csr = 0
        latency_load_delta_csr = 0
        latency_compute_delta_csr = 0
        latency_lif_delta_csr = 0
        latency_store_delta_csr = 0
        
        latency_a_delta_acsr = 0
        latency_load_delta_acsr = 0
        latency_compute_delta_acsr = 0
        latency_lif_delta_acsr = 0
        latency_store_delta_acsr = 0

        ops = 0

        T,B,C,H,W = x.shape
        global total_compute_latency
        global total_compute_latency_delta
        x = self.mlp1_lif(x)
        x_for_ltc = x
        delta_x1 = x_for_ltc[:,1,...] - x_for_ltc[:,0,...]
        # print(self.mlp1_conv.weight.shape) # o,i,h=1,w=1
        weight_for_ltc = self.mlp1_conv.weight.squeeze()
        latency_a_bl_temp, latency_load_bl_temp, latency_compute_bl_temp, latency_lif_bl_temp, latency_store_bl_temp = calculate_latency_for_mlp_lif_bl(x_for_ltc, weight_for_ltc, 1, 8)
        latency_a_bl += latency_a_bl_temp
        latency_load_bl += latency_load_bl_temp
        latency_compute_bl += latency_compute_bl_temp
        latency_lif_bl += latency_lif_bl_temp
        latency_store_bl += latency_store_bl_temp

        latency_a_delta_temp, latency_load_delta_temp, latency_compute_delta_temp, latency_lif_delta_temp = calculate_latency_for_mlp_lif_delta_i(delta_x1, weight_for_ltc, 8, 'COO')
        latency_a_delta += latency_a_delta_temp
        latency_load_delta += latency_load_delta_temp
        latency_compute_delta += latency_compute_delta_temp
        latency_lif_delta += latency_lif_delta_temp

        latency_a_delta_csr_temp, latency_load_delta_csr_temp, latency_compute_delta_csr_temp, latency_lif_delta_csr_temp = calculate_latency_for_mlp_lif_delta_i(delta_x1, weight_for_ltc, 8, 'CSR')
        latency_a_delta_csr += latency_a_delta_csr_temp
        latency_load_delta_csr += latency_load_delta_csr_temp
        latency_compute_delta_csr += latency_compute_delta_csr_temp
        latency_lif_delta_csr += latency_lif_delta_csr_temp

        latency_a_delta_acsr_temp, latency_load_delta_acsr_temp, latency_compute_delta_acsr_temp, latency_lif_delta_acsr_temp = calculate_latency_for_mlp_lif_delta_i(delta_x1, weight_for_ltc, 8, 'AdaptiveCSR')
        latency_a_delta_acsr += latency_a_delta_acsr_temp
        latency_load_delta_acsr += latency_load_delta_acsr_temp
        latency_compute_delta_acsr += latency_compute_delta_acsr_temp
        latency_lif_delta_acsr += latency_lif_delta_acsr_temp
        
        if timewindow == 3:
            delta_x2 = x_for_ltc[:,2,...] - x_for_ltc[:,1,...]
            latency_a_delta_temp, latency_load_delta_temp, latency_compute_delta_temp, latency_lif_delta_temp = calculate_latency_for_mlp_lif_delta_i(delta_x2, weight_for_ltc, 8, 'COO')
            latency_a_delta += latency_a_delta_temp
            latency_load_delta += latency_load_delta_temp
            latency_compute_delta += latency_compute_delta_temp
            latency_lif_delta += latency_lif_delta_temp
            
            latency_a_delta_csr_temp, latency_load_delta_csr_temp, latency_compute_delta_csr_temp, latency_lif_delta_csr_temp = calculate_latency_for_mlp_lif_delta_i(delta_x2, weight_for_ltc, 8, 'CSR')
            latency_a_delta_csr += latency_a_delta_csr_temp
            latency_load_delta_csr += latency_load_delta_csr_temp
            latency_compute_delta_csr += latency_compute_delta_csr_temp
            latency_lif_delta_csr += latency_lif_delta_csr_temp

            latency_a_delta_acsr_temp, latency_load_delta_acsr_temp, latency_compute_delta_acsr_temp, latency_lif_delta_acsr_temp = calculate_latency_for_mlp_lif_delta_i(delta_x2, weight_for_ltc, 8, 'AdaptiveCSR')
            latency_a_delta_acsr += latency_a_delta_acsr_temp
            latency_load_delta_acsr += latency_load_delta_acsr_temp
            latency_compute_delta_acsr += latency_compute_delta_acsr_temp
            latency_lif_delta_acsr += latency_lif_delta_acsr_temp

        # ops
        x_for_ops = x
        lif_ops = x_for_ops.numel()
        mlp_ops = 2 * self.mlp1_conv.weight.shape[0] * self.mlp1_conv.weight.shape[1] * x_for_ops.shape[0] * x_for_ops.shape[1] * x_for_ops.shape[-1] * x_for_ops.shape[-2]
        ops += lif_ops + mlp_ops

        # total_compute_latency += latency_a
        # total_compute_latency_delta += latency_delta
        x = self.mlp1_conv(x.flatten(0,1))
        x = self.mlp1_bn(x).reshape(T,B,self.c_hidden,H,W).contiguous()
        

        x = self.mlp2_lif(x)
        x_for_ltc = x
        delta_x1 = x_for_ltc[:,1,...] - x_for_ltc[:,0,...]
        weight_for_ltc = self.mlp2_conv.weight.squeeze() # weight shape: o, i, kh, kw
        latency_store_delta_temp = calculate_latency_for_mlp_lif_delta_o(delta_x1, 'COO')
        latency_store_delta += latency_store_delta_temp
        latency_store_delta_csr_temp = calculate_latency_for_mlp_lif_delta_o(delta_x1, 'CSR')
        latency_store_delta_csr += latency_store_delta_csr_temp
        latency_store_delta_acsr_temp = calculate_latency_for_mlp_lif_delta_o(delta_x1, 'AdaptiveCSR')
        latency_store_delta_acsr += latency_store_delta_acsr_temp
        if timewindow == 3:
            delta_x2 = x_for_ltc[:,2,...] - x_for_ltc[:,1,...]
            latency_store_delta_temp = calculate_latency_for_mlp_lif_delta_o(delta_x2, 'COO')
            latency_store_delta += latency_store_delta_temp
            latency_store_delta_csr_temp = calculate_latency_for_mlp_lif_delta_o(delta_x2, 'CSR')
            latency_store_delta_csr += latency_store_delta_csr_temp
            latency_store_delta_acsr_temp = calculate_latency_for_mlp_lif_delta_o(delta_x2, 'AdaptiveCSR')
            latency_store_delta_acsr += latency_store_delta_acsr_temp

        # total_compute_latency_delta += latency_delta
        latency_a_bl_temp, latency_load_bl_temp, latency_compute_bl_temp, latency_lif_bl_temp, latency_store_bl_temp = calculate_latency_for_mlp_lif_bl(x_for_ltc, weight_for_ltc, 1, 8)
        latency_a_bl += latency_a_bl_temp
        latency_load_bl += latency_load_bl_temp
        latency_compute_bl += latency_compute_bl_temp
        latency_lif_bl += latency_lif_bl_temp
        latency_store_bl += latency_store_bl_temp

        latency_a_delta_temp, latency_load_delta_temp, latency_compute_delta_temp, latency_lif_delta_temp = calculate_latency_for_mlp_lif_delta_i(delta_x1, weight_for_ltc, 8, 'COO')
        latency_a_delta += latency_a_delta_temp
        latency_load_delta += latency_load_delta_temp
        latency_compute_delta += latency_compute_delta_temp
        latency_lif_delta += latency_lif_delta_temp
        
        latency_a_delta_csr_temp, latency_load_delta_csr_temp, latency_compute_delta_csr_temp, latency_lif_delta_csr_temp = calculate_latency_for_mlp_lif_delta_i(delta_x1, weight_for_ltc, 8, 'CSR')
        latency_a_delta_csr += latency_a_delta_csr_temp
        latency_load_delta_csr += latency_load_delta_csr_temp
        latency_compute_delta_csr += latency_compute_delta_csr_temp
        latency_lif_delta_csr += latency_lif_delta_csr_temp

        latency_a_delta_acsr_temp, latency_load_delta_acsr_temp, latency_compute_delta_acsr_temp, latency_lif_delta_acsr_temp = calculate_latency_for_mlp_lif_delta_i(delta_x1, weight_for_ltc, 8, 'AdaptiveCSR')
        latency_a_delta_acsr += latency_a_delta_acsr_temp
        latency_load_delta_acsr += latency_load_delta_acsr_temp
        latency_compute_delta_acsr += latency_compute_delta_acsr_temp
        latency_lif_delta_acsr += latency_lif_delta_acsr_temp
        if timewindow == 3:
            latency_a_delta_temp, latency_load_delta_temp, latency_compute_delta_temp, latency_lif_delta_temp = calculate_latency_for_mlp_lif_delta_i(delta_x2, weight_for_ltc, 8, 'COO')
            latency_a_delta += latency_a_delta_temp
            latency_load_delta += latency_load_delta_temp
            latency_compute_delta += latency_compute_delta_temp
            latency_lif_delta += latency_lif_delta_temp

            latency_a_delta_csr_temp, latency_load_delta_csr_temp, latency_compute_delta_csr_temp, latency_lif_delta_csr_temp = calculate_latency_for_mlp_lif_delta_i(delta_x2, weight_for_ltc, 8, 'CSR')
            latency_a_delta_csr += latency_a_delta_csr_temp
            latency_load_delta_csr += latency_load_delta_csr_temp
            latency_compute_delta_csr += latency_compute_delta_csr_temp
            latency_lif_delta_csr += latency_lif_delta_csr_temp

            latency_a_delta_acsr_temp, latency_load_delta_acsr_temp, latency_compute_delta_acsr_temp, latency_lif_delta_acsr_temp = calculate_latency_for_mlp_lif_delta_i(delta_x2, weight_for_ltc, 8, 'AdaptiveCSR')
            latency_a_delta_acsr += latency_a_delta_acsr_temp
            latency_load_delta_acsr += latency_load_delta_acsr_temp
            latency_compute_delta_acsr += latency_compute_delta_acsr_temp
            latency_lif_delta_acsr += latency_lif_delta_acsr_temp

        # ops
        x_for_ops = x
        lif_ops = x_for_ops.numel()
        mlp_ops = 2 * self.mlp2_conv.weight.shape[0] * self.mlp2_conv.weight.shape[1] * x_for_ops.shape[0] * x_for_ops.shape[1] * x_for_ops.shape[-1] * x_for_ops.shape[-2]
        ops += lif_ops + mlp_ops

        # total_compute_latency += latency_a
        # total_compute_latency_delta += latency_delta
        x = self.mlp2_conv(x.flatten(0,1))
        x = self.mlp2_bn(x).reshape(T,B,self.c_output,H,W).contiguous()
        x_for_ltc = self.mlp1_lif(x)
        delta_x1 = x_for_ltc[:,1,...] - x_for_ltc[:,0,...]
        latency_store_delta_temp = calculate_latency_for_mlp_lif_delta_o(delta_x1, 'COO')
        latency_store_delta += latency_store_delta_temp
        latency_store_delta_csr_temp = calculate_latency_for_mlp_lif_delta_o(delta_x1, 'CSR')
        latency_store_delta_csr += latency_store_delta_csr_temp
        latency_store_delta_acsr_temp = calculate_latency_for_mlp_lif_delta_o(delta_x1, 'AdaptiveCSR')
        latency_store_delta_acsr += latency_store_delta_acsr_temp
        if timewindow == 3:
            delta_x2 = x_for_ltc[:,2,...] - x_for_ltc[:,1,...]
            latency_store_delta_temp = calculate_latency_for_mlp_lif_delta_o(delta_x2, 'COO')
            latency_store_delta += latency_store_delta_temp
            latency_store_delta_csr_temp = calculate_latency_for_mlp_lif_delta_o(delta_x2, 'CSR')
            latency_store_delta_csr += latency_store_delta_csr_temp
            latency_store_delta_acsr_temp = calculate_latency_for_mlp_lif_delta_o(delta_x2, 'AdaptiveCSR')
            latency_store_delta_acsr += latency_store_delta_acsr_temp
        # total_compute_latency_delta += latency_delta

        if timewindow == 3:
            ns_latency_a_bl = latency_a_bl * (1000/frequency)
            ns_latency_a_delta = (latency_a_delta+latency_a_bl/3) * (1000/frequency)
            ns_latency_a_delta_csr = (latency_a_delta_csr+latency_a_bl/3) * (1000/frequency)
            ns_latency_a_delta_acsr = (latency_a_delta_acsr+latency_a_bl/3) * (1000/frequency)
            ns_latency_ema_bl = (latency_load_bl + latency_store_bl) * (1000/frequency)
            ns_latency_ema_delta = ((latency_load_delta + latency_store_delta) + (latency_load_bl + latency_store_bl) / 3) * (1000/frequency)
            ns_latency_ema_delta_csr = ((latency_load_delta_csr + latency_store_delta_csr) + (latency_load_bl + latency_store_bl) / 3) * (1000/frequency)
            ns_latency_ema_delta_acsr = ((latency_load_delta_acsr + latency_store_delta_acsr) + (latency_load_bl + latency_store_bl) / 3) * (1000/frequency)
            ns_latency_compute_bl = (latency_compute_bl + latency_lif_bl) * (1000/frequency)
            ns_latency_compute_delta = ((latency_compute_delta + latency_lif_delta) + (latency_compute_bl + latency_lif_bl) / 3) * (1000/frequency)
        else:
            ns_latency_a_bl = latency_a_bl * (1000/frequency)
            ns_latency_a_delta = (latency_a_delta+latency_a_bl/2) * (1000/frequency)
            ns_latency_a_delta_csr = (latency_a_delta_csr+latency_a_bl/2) * (1000/frequency)
            ns_latency_a_delta_acsr = (latency_a_delta_acsr+latency_a_bl/2) * (1000/frequency)
            ns_latency_ema_bl = (latency_load_bl + latency_store_bl) * (1000/frequency)
            ns_latency_ema_delta = (latency_load_delta + latency_store_delta) * (1000/frequency)
            ns_latency_ema_delta_csr = (latency_load_delta_csr + latency_store_delta_csr) * (1000/frequency)
            ns_latency_ema_delta_acsr = (latency_load_delta_acsr + latency_store_delta_acsr) * (1000/frequency)
            ns_latency_compute_bl = (latency_compute_bl + latency_lif_bl) * (1000/frequency)
            ns_latency_compute_delta = (latency_compute_delta + latency_lif_delta + (latency_compute_bl + latency_lif_bl) / 2) * (1000/frequency)
        # print(f'MLP Latency: {ns_latency_a_bl} ns')
        # print(f'MLP EMA Latency: {ns_latency_ema_bl} ns')
        # print(f'MLP Compute Latency: {ns_latency_compute_bl} ns')

        # print(f'MLP Delta Latency: {ns_latency_a_delta} ns')
        # print(f'MLP Delta EMA Latency: {ns_latency_ema_delta} ns')
        # print(f'MLP Delta Compute Latency: {ns_latency_compute_delta} ns')

        # print(f'The performance boost Delta brings in MLP: {ns_latency_a_delta/ns_latency_a_bl}')
        # print(f'The performance boost Delta brings in MLP, EMA: {ns_latency_ema_delta/ns_latency_ema_bl}')

        # print(f'MLP Delta Latency CSR: {ns_latency_a_delta_csr} ns')
        # print(f'MLP Delta EMA Latency CSR: {ns_latency_ema_delta_csr} ns')

        # print(f'The performance boost Delta+CSR brings in MLP: {ns_latency_a_delta_csr/ns_latency_a_bl}')
        # print(f'The performance boost Delta+CSR brings in MLP, EMA: {ns_latency_ema_delta_csr/ns_latency_ema_bl}')

        # print(f'MLP Delta Latency AdaptiveCSR: {ns_latency_a_delta_acsr} ns')
        # print(f'MLP Delta EMA Latency AdaptiveCSR: {ns_latency_ema_delta_acsr} ns')

        # print(f'The performance boost Delta+AdaptiveCSR brings in MLP: {ns_latency_a_delta_acsr/ns_latency_a_bl}')
        # print(f'The performance boost Delta+AdaptiveCSR brings in MLP, EMA: {ns_latency_ema_delta_acsr/ns_latency_ema_bl}')

        # print(f'The performance boost in MLP, Compute: {ns_latency_compute_delta/ns_latency_compute_bl}')

        # # calculating OPs
        # TOPs = ops / 1e12
        # print(f'The total number of TOPs in MLP: {TOPs}')
        # TOPS_bl = TOPs / (ns_latency_a_bl * 1e-9)
        # TOPS_bl_compute_only = TOPs / (ns_latency_compute_bl * 1e-9)
        # TOPS_delta = TOPs / (ns_latency_a_delta * 1e-9)
        # TOPS_delta_compute_only = TOPs / (ns_latency_compute_delta * 1e-9)
        # print(f'The total number of TOPS in MLP: {TOPS_bl} TOPS')
        # print(f'The total number of TOPS in MLP, Compute: {TOPS_bl_compute_only} TOPS')
        # print(f'The total number of TOPS in MLP, Delta: {TOPS_delta} TOPS')
        # print(f'The total number of TOPS in MLP, Delta, Compute: {TOPS_delta_compute_only} TOPS')

        return x


class SpikingSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='torch')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)


    def forward(self, x):
        T,B,C,H,W = x.shape
        global total_compute_latency
        global total_compute_latency_delta
        x = self.proj_lif(x)
        
        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x_for_qkv_ltc = x_for_qkv.unsqueeze(1).transpose(-1,-2)
        weight_for_ltc = self.q_conv.weight.squeeze(2).unsqueeze(0).repeat(T,1,1,1)
        latency_qkvgen = calculate_latency_for_qkvgen_lif_bl(x_for_qkv_ltc, weight_for_ltc, 1, 8)
        total_compute_latency += latency_qkvgen
        # print(f'QKV generation Latency:{latency_qkvgen}')
        
        x = k.transpose(-2,-1) @ v
        latency_kvq=0
        latency_kvq += calculate_latency_for_qkvmm_nonlif_bl(k.transpose(-2,-1), v, 1, 1)
        # F.cosine_similarity(x[:,1,...].reshape(1,-1),x[:,0,...].reshape(1,-1))  #0.9746
        # F.cosine_similarity(x[:,1,...].reshape(1,-1),(x[:,0,...]+k_deltav).reshape(1,-1))   #0.9828
        # F.cosine_similarity(x[:,1,...].reshape(1,-1),(x[:,0,...]+deltak_v).reshape(1,-1))   #0.9866
        # F.cosine_similarity(x[:,1,...].reshape(1,-1),(x[:,0,...]+k_deltav+deltak_v).reshape(1,-1))  #0.9895
        # F.cosine_similarity(x[:,1,...].reshape(1,-1),(x[:,0,...]+k_deltav+deltak_v+delta_kv).reshape(1,-1)) #1.000

        x = (q @ x) * self.scale
        latency_kvq += calculate_latency_for_qkvmm_lif_bl(q, x, 1, 8)
        # total_compute_latency += latency_kvq
        # print(f'Attention Latency:{latency_kvq}')
        # pdb.set_trace()
        # compute_latency = calculate_latency_for_attn(q_conv_out, k_conv_out)
        # total_compute_latency += compute_latency
        # print('attention:', compute_latency, 'total:', total_compute_latency)

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)

        delta_x1 = x.transpose(-1, -2)[:,1,...] - x.transpose(-1, -2)[:,0,...]
        if timewindow == 3:
            delta_x2 = x.transpose(-1, -2)[:,2,...] - x.transpose(-1, -2)[:,1,...]
        x = x.flatten(0,1)
        x_for_ltc = x.unsqueeze(1).transpose(-1,-2)
        weight_for_ltc = self.proj_conv.weight.squeeze(2)

        latency_proj_a_delta = 0
        latency_proj_load_delta = 0
        latency_proj_compute_delta = 0
        latency_proj_lif_delta = 0
        latency_proj_store_delta = 0
        latency_proj_a_delta_csr = 0
        latency_proj_load_delta_csr = 0
        latency_proj_compute_delta_csr = 0
        latency_proj_lif_delta_csr = 0
        latency_proj_store_delta_csr = 0
        latency_proj_a_delta_acsr = 0
        latency_proj_load_delta_acsr = 0
        latency_proj_compute_delta_acsr = 0
        latency_proj_lif_delta_acsr = 0
        latency_proj_store_delta_acsr = 0

        latency_proj_a_bl, latency_proj_load_bl, latency_proj_compute_bl, latency_proj_lif_bl, latency_proj_store_bl = calculate_latency_for_proj_lif_bl(x_for_ltc, weight_for_ltc, 1, 8)
        latency_proj_a_delta_temp, latency_proj_load_delta_temp, latency_proj_compute_delta_temp, latency_proj_lif_delta_temp = calculate_latency_for_proj_lif_delta_i(delta_x1, weight_for_ltc, 8, 'COO')
        latency_proj_a_delta += latency_proj_a_delta_temp
        latency_proj_load_delta += latency_proj_load_delta_temp
        latency_proj_compute_delta += latency_proj_compute_delta_temp
        latency_proj_lif_delta += latency_proj_lif_delta_temp
        latency_proj_a_delta_csr_temp, latency_proj_load_delta_csr_temp, latency_proj_compute_delta_csr_temp, latency_proj_lif_delta_csr_temp = calculate_latency_for_proj_lif_delta_i(delta_x1, weight_for_ltc, 8, 'CSR')
        latency_proj_a_delta_csr += latency_proj_a_delta_csr_temp
        latency_proj_load_delta_csr += latency_proj_load_delta_csr_temp
        latency_proj_compute_delta_csr += latency_proj_compute_delta_csr_temp
        latency_proj_lif_delta_csr += latency_proj_lif_delta_csr_temp
        latency_proj_a_delta_acsr_temp, latency_proj_load_delta_acsr_temp, latency_proj_compute_delta_acsr_temp, latency_proj_lif_delta_acsr_temp = calculate_latency_for_proj_lif_delta_i(delta_x1, weight_for_ltc, 8, 'AdaptiveCSR')
        latency_proj_a_delta_acsr += latency_proj_a_delta_acsr_temp
        latency_proj_load_delta_acsr += latency_proj_load_delta_acsr_temp
        latency_proj_compute_delta_acsr += latency_proj_compute_delta_acsr_temp
        latency_proj_lif_delta_acsr += latency_proj_lif_delta_acsr_temp

        if timewindow == 3:
            latency_proj_a_delta_temp, latency_proj_load_delta_temp, latency_proj_compute_delta_temp, latency_proj_lif_delta_temp = calculate_latency_for_proj_lif_delta_i(delta_x2, weight_for_ltc, 8, 'COO')
            latency_proj_a_delta += latency_proj_a_delta_temp
            latency_proj_load_delta += latency_proj_load_delta_temp
            latency_proj_compute_delta += latency_proj_compute_delta_temp
            latency_proj_lif_delta += latency_proj_lif_delta_temp
            latency_proj_a_delta_csr_temp, latency_proj_load_delta_csr_temp, latency_proj_compute_delta_csr_temp, latency_proj_lif_delta_csr_temp = calculate_latency_for_proj_lif_delta_i(delta_x2, weight_for_ltc, 8, 'CSR')
            latency_proj_a_delta_csr += latency_proj_a_delta_csr_temp
            latency_proj_load_delta_csr += latency_proj_load_delta_csr_temp
            latency_proj_compute_delta_csr += latency_proj_compute_delta_csr_temp
            latency_proj_lif_delta_csr += latency_proj_lif_delta_csr_temp
            latency_proj_a_delta_acsr_temp, latency_proj_load_delta_acsr_temp, latency_proj_compute_delta_acsr_temp, latency_proj_lif_delta_acsr_temp = calculate_latency_for_proj_lif_delta_i(delta_x2, weight_for_ltc, 8, 'AdaptiveCSR')
            latency_proj_a_delta_acsr += latency_proj_a_delta_acsr_temp
            latency_proj_load_delta_acsr += latency_proj_load_delta_acsr_temp
            latency_proj_compute_delta_acsr += latency_proj_compute_delta_acsr_temp
            latency_proj_lif_delta_acsr += latency_proj_lif_delta_acsr_temp
        # total_compute_latency += latency_proj
        
        # ops
        ops = 0
        x_for_ops = x
        lif_ops = x_for_ops.numel()
        mlp_ops = 2 * self.proj_conv.weight.shape[0] * self.proj_conv.weight.shape[1] * x_for_ops.shape[0] * x_for_ops.shape[-1]
        ops += lif_ops + mlp_ops

        x = self.proj_bn(self.proj_conv(x)).reshape(T,B,C,H,W)
        x_for_ltc = self.proj_lif(x)
        delta_x1 = x_for_ltc[:,1,...] - x_for_ltc[:,0,...]
        latency_proj_store_delta += calculate_latency_for_mlp_lif_delta_o(delta_x1, 'COO')
        latency_proj_store_delta_csr += calculate_latency_for_mlp_lif_delta_o(delta_x1, 'CSR')
        latency_proj_store_delta_acsr += calculate_latency_for_mlp_lif_delta_o(delta_x1, 'AdaptiveCSR')
        if timewindow == 3:
            delta_x2 = x_for_ltc[:,2,...] - x_for_ltc[:,1,...]
            latency_proj_store_delta += calculate_latency_for_mlp_lif_delta_o(delta_x2, 'COO')
            latency_proj_store_delta_csr += calculate_latency_for_mlp_lif_delta_o(delta_x2, 'CSR')
            latency_proj_store_delta_acsr += calculate_latency_for_mlp_lif_delta_o(delta_x2, 'AdaptiveCSR')

        if timewindow == 3:
            ns_latency_proj_a_bl = latency_proj_a_bl * (1000/frequency)
            ns_latency_proj_a_delta = (latency_proj_a_delta+latency_proj_a_bl/3) * (1000/frequency)
            ns_latency_proj_a_delta_csr = (latency_proj_a_delta_csr+latency_proj_a_bl/3) * (1000/frequency)
            ns_latency_proj_a_delta_acsr = (latency_proj_a_delta_acsr+latency_proj_a_bl/3) * (1000/frequency)
            ns_latency_proj_ema_bl = (latency_proj_load_bl + latency_proj_store_bl) * (1000/frequency)
            ns_latency_proj_ema_delta = ((latency_proj_load_delta + latency_proj_store_delta) + (latency_proj_load_bl + latency_proj_store_bl) / 3) * (1000/frequency)
            ns_latency_proj_ema_delta_csr = ((latency_proj_load_delta_csr + latency_proj_store_delta_csr) + (latency_proj_load_bl + latency_proj_store_bl) / 3) * (1000/frequency)
            ns_latency_proj_ema_delta_acsr = ((latency_proj_load_delta_acsr + latency_proj_store_delta_acsr) + (latency_proj_load_bl + latency_proj_store_bl) / 3) * (1000/frequency)
            ns_latency_proj_compute_bl = (latency_proj_compute_bl + latency_proj_lif_bl) * (1000/frequency)
            ns_latency_proj_compute_delta = ((latency_proj_compute_delta + latency_proj_lif_delta) + (latency_proj_compute_bl + latency_proj_lif_bl) / 3) * (1000/frequency)

        else:
            ns_latency_proj_a_bl = latency_proj_a_bl * (1000/frequency)
            ns_latency_proj_a_delta = (latency_proj_a_delta+latency_proj_a_bl/2) * (1000/frequency)
            ns_latency_proj_a_delta_csr = (latency_proj_a_delta_csr+latency_proj_a_bl/2) * (1000/frequency)
            ns_latency_proj_a_delta_acsr = (latency_proj_a_delta_acsr+latency_proj_a_bl/2) * (1000/frequency)
            ns_latency_proj_ema_bl = (latency_proj_load_bl + latency_proj_store_bl) * (1000/frequency)
            ns_latency_proj_ema_delta = ((latency_proj_load_delta + latency_proj_store_delta) + (latency_proj_load_bl + latency_proj_store_bl) / 2) * (1000/frequency)
            ns_latency_proj_ema_delta_csr = ((latency_proj_load_delta_csr + latency_proj_store_delta_csr) + (latency_proj_load_bl + latency_proj_store_bl) / 2) * (1000/frequency)
            ns_latency_proj_ema_delta_acsr = ((latency_proj_load_delta_acsr + latency_proj_store_delta_acsr) + (latency_proj_load_bl + latency_proj_store_bl) / 2) * (1000/frequency)
            ns_latency_proj_compute_bl = (latency_proj_compute_bl + latency_proj_lif_bl) * (1000/frequency)
            ns_latency_proj_compute_delta = (latency_proj_compute_delta + latency_proj_lif_delta + (latency_proj_compute_bl + latency_proj_lif_bl) / 2) * (1000/frequency)

        # print(f'Projection in Attn Block Latency: {ns_latency_proj_a_bl} ns')
        # print(f'Projection in Attn Block EMA Latency: {ns_latency_proj_ema_bl} ns')
        # print(f'Projection in Attn Block Compute Latency: {ns_latency_proj_compute_bl} ns')

        # print(f'Projection in Attn Block Delta Latency: {ns_latency_proj_a_delta} ns')
        # print(f'Projection in Attn Block Delta EMA Latency: {ns_latency_proj_ema_delta} ns')
        # print(f'Projection in Attn Block Delta Compute Latency: {ns_latency_proj_compute_delta} ns')

        # print(f'The performance boost Delta brings in Projection: {ns_latency_proj_a_delta/ns_latency_proj_a_bl}')
        # print(f'The performance boost Delta brings in Projection, EMA: {ns_latency_proj_ema_delta/ns_latency_proj_ema_bl}')

        # print(f'Projection in Attn Block Delta Latency CSR: {ns_latency_proj_a_delta_csr} ns')
        # print(f'Projection in Attn Block Delta EMA Latency CSR: {ns_latency_proj_ema_delta_csr} ns')

        # print(f'The performance boost Delta+CSR brings in Projection: {ns_latency_proj_a_delta_csr/ns_latency_proj_a_bl}')
        # print(f'The performance boost Delta+CSR brings in Projection, EMA: {ns_latency_proj_ema_delta_csr/ns_latency_proj_ema_bl}')

        # print(f'Projection in Attn Block Delta Latency AdaptiveCSR: {ns_latency_proj_a_delta_acsr} ns')
        # print(f'Projection in Attn Block Delta EMA Latency AdaptiveCSR: {ns_latency_proj_ema_delta_acsr} ns')

        # print(f'The performance boost Delta+AdaptiveCSR brings in Projection: {ns_latency_proj_a_delta_acsr/ns_latency_proj_a_bl}')
        # print(f'The performance boost Delta+AdaptiveCSR brings in Projection, EMA: {ns_latency_proj_ema_delta_acsr/ns_latency_proj_ema_bl}')

        # print(f'The performance boost in Projection, Compute: {ns_latency_proj_compute_delta/ns_latency_proj_compute_bl}')

        # # calculating OPs
        # TOPs = ops / 1e12
        # print(f'The total number of TOPs in Projection: {TOPs}')
        # TOPS_bl = TOPs / (ns_latency_proj_a_bl * 1e-9)
        # TOPS_bl_compute_only = TOPs / (ns_latency_proj_compute_bl * 1e-9)
        # TOPS_delta = TOPs / (ns_latency_proj_a_delta * 1e-9)
        # TOPS_delta_compute_only = TOPs / (ns_latency_proj_compute_delta * 1e-9)
        # print(f'The total number of TOPS in Projection: {TOPS_bl} TOPS')
        # print(f'The total number of TOPS in Projection, Compute: {TOPS_bl_compute_only} TOPS')
        # print(f'The total number of TOPS in Projection, Delta: {TOPS_delta} TOPS')
        # print(f'The total number of TOPS in Projection, Delta, Compute: {TOPS_delta_compute_only} TOPS')

        return x, x_for_qkv.reshape(T,B,C,N).contiguous(), q_conv_out, k_conv_out, v_conv_out

class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpikingSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x_attn, x_for_qkv, q_conv_out, k_conv_out, v_conv_out = self.attn(x)
        x = x + x_attn
        x = self.mlp(x)

        return x, x_for_qkv, q_conv_out, k_conv_out, v_conv_out


class SpikingTokenizer(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)

        self.proj1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj1_conv = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims//4)

        self.proj2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj2_conv = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims//2)

        self.proj3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj3_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)

        self.proj4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        # global total_compute_latency
        T, B, C, H, W = x.shape
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()

        x = self.proj1_lif(x).flatten(0,1).contiguous()
        x = self.maxpool1(x)
        # compute_latency = calculate_latency_for_conv(x,self.proj1_conv.out_channels)
        # total_compute_latency += compute_latency
        # print('SPS_conv1:',compute_latency,'total:',total_compute_latency)
        x = self.proj1_conv(x)
        x = self.proj1_bn(x).reshape(T, B, -1, H//2, W//2).contiguous()
        
        x = self.proj2_lif(x).flatten(0, 1).contiguous()
        x = self.maxpool2(x)
        # compute_latency = calculate_latency_for_conv(x,self.proj2_conv.out_channels)
        # total_compute_latency += compute_latency
        # print('SPS_conv2:',compute_latency,'total:',total_compute_latency)
        x = self.proj2_conv(x)
        x = self.proj2_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()

        x = self.proj3_lif(x).flatten(0, 1).contiguous()
        x = self.maxpool3(x)
        # compute_latency = calculate_latency_for_conv(x,self.proj3_conv.out_channels)
        # total_compute_latency += compute_latency
        # print('SPS_conv3:',compute_latency,'total:',total_compute_latency)
        x = self.proj3_conv(x)
        x = self.proj3_bn(x).reshape(T, B, -1, H//8, W//8).contiguous()

        x = self.proj4_lif(x).flatten(0, 1).contiguous()
        x = self.maxpool4(x)
        # compute_latency = calculate_latency_for_conv(x,self.proj4_conv.out_channels)
        # total_compute_latency += compute_latency
        # print('SPS_conv4:',compute_latency,'total:',total_compute_latency)
        x = self.proj4_conv(x)
        x = self.proj4_bn(x).reshape(T, B, -1, H//16, W//16).contiguous()

        # print(embed_sparsity)

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)

class vit_snn(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T = 4, pretrained_cfg= None
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SpikingTokenizer(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims)

        block = nn.ModuleList([SpikingTransformer(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        # print("Tokenizing...", file=open("sparsity.txt", "a"))
        x, (H, W) = patch_embed(x)
        attn = None
        for block_id, blk in enumerate(block):
            # print('ATTENTION BLOCK', block_id, file=open("sparsity.txt", "a"))
            x, x_for_qkv, q, k, v = blk(x)
            # if save:
            #     save_path = './statistics/imagenet/'+'block_'+str(block_id)+'_x.pth'
            #     torch.save(x_for_qkv, save_path)
            #     save_path = './statistics/imagenet/'+'block_'+str(block_id)+'_q.pth'
            #     torch.save(q, save_path)
            #     save_path = './statistics/imagenet/'+'block_'+str(block_id)+'_k.pth'
            #     torch.save(k, save_path)
            #     save_path = './statistics/imagenet/'+'block_'+str(block_id)+'_v.pth'
            #     torch.save(v, save_path)   
        # print(x.shape)                                             
        # return x.flatten(3).mean(3)
        return x.flatten(3)

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        # total_compute_latency_delta += latency_proj_delta
        # print(f'Total OPS: {total_OPS/1e12} TOPS')

        # print(f'Energy Efficiency: {total_OPS / physical_latency / power * 1e-9} TOPS/W')        
        # task_energy = ((power * 1e-3) * (total_compute_latency * (1000/frequency) * 1e-9)) * 1e3 #mJ
        # pJ_SOP = task_energy / total_spike_all * 1e9
        # print(f'Task energy: {task_energy} mJ, {pJ_SOP} pJ/SOP')
        # global ema
        # ema = ema / 1024/1024
        # print('ema spike:',ema[0],'ema weight',ema[1],'ema psum',ema[2])
        # print('BASELINE: ema spike:',ema[0],'ema weight',ema[1] * 8,'ema psum',ema[3])
        # x = self.head(x.mean(0))
        x = self.head(x.mean(3).mean(0))
        # total_compute_latency_delta += latency_proj_delta

        # print(f'Overall latency:{total_compute_latency}')
        return x


@register_model
def Spikingformer(pretrained=False, **kwargs):
    model = vit_snn(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

from timm.models import create_model

if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224).cuda()
    model = create_model(
        'Spikingformer',
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=512, num_heads=8, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=8, sr_ratios=1,
        T = 4
    ).cuda()

    model.eval()
    y = model(x)
    print(y.shape)
    print('Test Good!')
