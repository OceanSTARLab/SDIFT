
import numpy as np
import torch
import random

import time


extensions = ['*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.bmp']

#----------------------------------------------------------------------------
# EDM sampler & EDM model

def set_seed(seed: int = 42):
    random.seed(seed)                      # Python 内置随机库
    np.random.seed(seed)                   # NumPy
    torch.manual_seed(seed)                # PyTorch CPU
    torch.cuda.manual_seed(seed)           # 单个 GPU
    torch.cuda.manual_seed_all(seed)       # 多个 GPU（可选）

    torch.backends.cudnn.deterministic = True  # 保证卷积等操作可复现
    torch.backends.cudnn.benchmark = False     # 固定算法路径（可能稍慢）


def get_observations(ob_path):
    d = np.load(ob_path, allow_pickle=True).item() # time, lat, lon, depth
    d = d["data"]
    time_uni = d["t_ind_uni"]
    ob_y = d["ob_y"]
    ob_ind_conti = d["ob_ind_conti"] 
    ob_conti = ob_ind_conti[:, 1:]
    ob_ind = d["ob_ind"] 
    ob_time_ind = ob_ind[:,0]

    y_group = []
    ind_conti_group = []
    y_time_group = []
    y_time_ind_group = []
    for i in range(time_uni.shape[0]):
        y_temp = ob_y[ob_time_ind==i]
        y_group.append(y_temp+0*np.random.randn(*y_temp.shape))
        ind_conti_group.append(ob_conti[ob_time_ind==i])
        y_time_group.append(time_uni[i])
        y_time_ind_group.append(i)

    return y_group, ind_conti_group, y_time_group, y_time_ind_group


def get_ktT(y_tt, core_t, gp_gamma=5, gp_sigma=1):
    r = torch.sqrt(torch.square(y_tt - core_t)) #  time differences, shape [B, S]

    return gp_sigma*torch.exp(-torch.square(r)*gp_gamma) # [B, S]
    #return matern_kernel(r, sigma_f=1.0, l=1.0, nu=2.5)


def get_kTT_inv(t, gp_gamma=5, gp_sigma=1):
    r = torch.sqrt(torch.square(t - t.transpose(-1, -2))) # Pairwise time differences, shape [B, S, S]
    diag = torch.eye(t.shape[-2]).to(t) * 1e-3 # for numerical stability 100 * 100

    K = gp_sigma*torch.exp(-torch.square(r)*gp_gamma) + diag
    #L = torch.linalg.cholesky(K.squeeze(0))

    #return (L.T@L).unsqueeze(0)
    return torch.inverse(K)
    #return torch.inverse(matern_kernel(r, sigma_f=1.0, l=1.0, nu=2.5) + diag)
    



def create_mask(shape, r):
    # 计算1的个数
    num_ones = int(np.prod(shape)*(r))
    
    # 创建一个包含1和0的数组
    mask = np.zeros(np.prod(shape), dtype=int)
    
    # 随机选择位置填充1
    ones_indices = np.random.choice(np.prod(shape), num_ones, replace=False)
    mask[ones_indices] = 1
    # 将一维数组重塑为二维矩阵
    mask = mask.reshape(shape)
    return mask

def create_mask2(shape, r, unbalance_c=0.6): #non-uniform sampling
    # 计算1的个数
    num_ones = int(np.prod(shape)*(r))

    shape_split = (shape[0], shape[1], shape[2], int(shape[-1]/2)) # 分成两个mask
    
    # 创建一个包含1和0的数组
    mask1 = np.zeros(np.prod(shape_split), dtype=int)
    mask2 = np.zeros(np.prod(shape_split), dtype=int)


    # 随机选择位置填充1
    ones_indices1 = np.random.choice(np.prod(shape_split), int(unbalance_c*num_ones), replace=False)
    ones_indices2 = np.random.choice(np.prod(shape_split), num_ones-int(unbalance_c*num_ones), replace=False)
    mask1[ones_indices1] = 1
    mask2[ones_indices2] = 1
    # 将一维数组重塑为二维矩阵
    mask1 = mask1.reshape(shape_split)
    mask2 = mask2.reshape(shape_split)
    mask = np.concatenate((mask1, mask2), axis=-1) # [T, I1, I2, I3]
    return mask

def get_te_observations(d, rho=0.02, mode=1, ind=0):

    d_shape = d.shape # [T, I1, I2, I3]

    data_extract = d[ind]
    mask = create_mask(data_extract.shape, rho)
    if mode == 1:
        pass
    elif mode == 2:
        mask[1::2, :, :, :] = 0 # 间隔采样
    

    # extract samples
    t_ind_uni = np.linspace(0, d_shape[1]-1, d_shape[1]).astype(int)/(d_shape[1]-1)
    if d_shape[2] == 1:
        u_ind_uni = np.ones_like([1])
    else:
        u_ind_uni = np.linspace(0, d_shape[2]-1, d_shape[2]).astype(int)/(d_shape[2]-1)
    v_ind_uni = np.linspace(0, d_shape[3]-1, d_shape[3]).astype(int)/(d_shape[3]-1)
    w_ind_uni = np.linspace(0, d_shape[4]-1, d_shape[4]).astype(int)/(d_shape[4]-1)

    indices_ob = np.where(mask == 1) #提取观测数据的索引
    ob_ind_conti = np.array([t_ind_uni[indices_ob[0]], u_ind_uni[indices_ob[1]], v_ind_uni[indices_ob[2]], w_ind_uni[indices_ob[3]]]).T
    ob_ind = np.array(indices_ob).T
    ob_y = data_extract[indices_ob]

    ob_time_ind = ob_ind[:,0]
    ob_conti = ob_ind_conti[:, 1:]


    y_group = []
    ind_conti_group = []
    y_time_group = []
    y_time_ind_group = []
    for i in range(t_ind_uni.shape[0]):
        y_temp = ob_y[ob_time_ind==i]
        y_group.append(y_temp+0*np.random.randn(*y_temp.shape))
        ind_conti_group.append(ob_conti[ob_time_ind==i])
        y_time_group.append(t_ind_uni[i])
        y_time_ind_group.append(i)

    return data_extract, mask, y_group, ind_conti_group, y_time_group, y_time_ind_group, u_ind_uni, v_ind_uni, w_ind_uni




def decoder(u_ind_uni, v_ind_uni,w_ind_uni, core, basis_function, device="cpu"):
    
    u_ind_uni = torch.FloatTensor(u_ind_uni).to(device)
    v_ind_uni = torch.FloatTensor(v_ind_uni).to(device)
    w_ind_uni = torch.FloatTensor(w_ind_uni).to(device)

    ind_input = (u_ind_uni, v_ind_uni, w_ind_uni)

    basis_function.eval()
    basis_function.mode = "training"

    core = core.to(torch.float32)
    basises  = basis_function(input_ind_train = ind_input)  # (I1*R1, I2*R2, I3*R3)
    output = torch.einsum("mi, tijk->tmjk", basises[0], core)
    output = torch.einsum("nj, tmjk->tmnk", basises[1], output)
    output = torch.einsum("ok, tmnk->tmno", basises[2], output)

    output = output.cpu().detach().numpy()
    return output



if __name__ == "__main__":
    pass