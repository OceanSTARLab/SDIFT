import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import h5py


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)







def get_ind_time(data_dict, key1, key2):
    x1 = data_dict[key1]
    ind1 = x1[:, :2]
    t1 = x1[:, -1]
    x2 = data_dict[key2]
    ind2 = x2[:, :2]
    t2 = x2[:, -1]
    return ind1, ind2, t1, t2



def normalize_data(tr_y, te_y):
    data_mean = tr_y.mean()
    data_std = tr_y.std()
    tr = (tr_y - data_mean) / data_std
    te = (te_y - data_mean) / data_std

    return tr, te, data_std, data_mean



def normalize_data2(d):
    min_val = d.min()
    max_val = d.max()


    return 2 * (d - min_val) / (max_val - min_val) - 1

def get_sample_data(tr_time_ind):
    values = np.array([4*x for x in range(67)])
    mask = np.isin(tr_time_ind, values)
    indices = np.where(mask)
    return indices[0]



def create_dict(list):
    my_dict = {x: list[x] for x in range(len(list))}
    return my_dict


def create_dict2(list):
    my_dict = {list[x]:x for x in range(len(list))}
    return my_dict




def load_data_ssf(data_path, flag="train"):
    if flag == "train":
        d = np.load(data_path, allow_pickle=True).item() # time, lat, lon, depth
        dims = d["ndims"]
        d = d["data"]
        tr_ind_conti = d["tr_ind_conti"]
        tr_ind = d["tr_ind"]
        tr_time_ind = tr_ind[:, 0]
        tr_dep_ind = tr_ind[:, -1]
        tr_ind = tr_ind[:, 1:3]
        tr_y = d["tr_y"]
        u_ind_uni = d["u_ind_uni"]
        v_ind_uni = d["v_ind_uni"]
        return tr_ind_conti, tr_ind, tr_dep_ind, tr_time_ind, tr_y,   u_ind_uni, v_ind_uni, dims
    elif flag == "test":
        d = np.load(data_path, allow_pickle=True).item()   
        d = d["data"]
        te_ind_conti = d["te_ind_conti"]
        te_ind = d["te_ind"]
        te_time_ind = te_ind[:, 0]
        te_dep_ind = te_ind[:, -1]
        te_ind = te_ind[:, 1:3]
        te_y = d["te_y"]
        return te_ind_conti, te_ind, te_dep_ind, te_time_ind, te_y



def load_data_se(data_path):
    d = np.load(data_path, allow_pickle=True).item() # time, lat, lon, depth
    d = d["data"]
    ind_uni = (d["u_ind_uni"], d["v_ind_uni"], d["w_ind_uni"], d["t_ind_uni"])
    data_extract = d["data"]
    mask_tr = d["mask_tr"]
    data_mean = d["data_mean"]
    data_std = d["data_std"]  
    return ind_uni, data_extract, mask_tr, data_mean, data_std


def load_large_data(data_path, metadata_path):
    with h5py.File(data_path, 'r') as f:
        data_extract = f['data'][:]
    d = np.load(metadata_path, allow_pickle=True).item()
    d = d["data"]
    ind_uni = (d["u_ind_uni"], d["v_ind_uni"], d["w_ind_uni"], d["t_ind_uni"])
    mask_tr = d["mask_tr"]
    return ind_uni, data_extract, mask_tr


def plot_training_data(tr_ind, tr_dep_ind, tr_time_ind, tr_y):
    a = np.zeros((38, 76, 4))
   
    for i in range(tr_y.shape[0]):
        if tr_dep_ind[i] == 0:
            timestep = tr_time_ind[i]
            if timestep == 0:
                a[tr_ind[i][0], tr_ind[i][1], 0] = tr_y[i]
            elif timestep == 10:
                a[tr_ind[i][0], tr_ind[i][1], 1] = tr_y[i]
            elif timestep == 20:
                a[tr_ind[i][0], tr_ind[i][1], 2] = tr_y[i]
            elif timestep == 30:
                a[tr_ind[i][0], tr_ind[i][1], 3] = tr_y[i]

    vmin = tr_y.min()
    vmax = tr_y.max()

    # 创建一个图形和两个子图
    fig, axs = plt.subplots(2, 2, figsize=(5, 4))
    cmap = plt.cm.viridis
    cmap.set_under('white')
    # 可视化第一个矩阵
    cax1 = axs[0,0].imshow(np.squeeze(a[:,:,0]), cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    axs[0,0].set_title('t=0')
    cax1 = axs[0,1].imshow(np.squeeze(a[:,:,1]), cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    axs[0,1].set_title('t=10')
    cax1 = axs[1,0].imshow(np.squeeze(a[:,:,2]), cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    axs[1,0].set_title('t=20')
    cax1 = axs[1,1].imshow(np.squeeze(a[:,:,3]), cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    axs[1,1].set_title('t=30')



    # 展示图形
    plt.tight_layout()
    plt.show()
###########################################################################




###################################################################
def tv_regularization(tensor, weight=1.0, flag = True):
    diff = tensor[:-1] - tensor[1:]
    if flag:
        tv_loss = torch.sum(torch.norm(diff, p='fro', dim=(1, 2)))
    else:
        tv_loss = torch.sum(torch.norm(diff, p='fro', dim=(1)))
    return weight * tv_loss



def total_variation_loss(X, weight):
    # 计算相邻元素的差异
    diff = X[:, 1:, :, :, :] - X[:, :-1, :, :, :]
    tv_loss = torch.sum(torch.norm(diff, p='fro', dim=(1)))
    return weight * tv_loss

def get_gp_covariance(t, gp_gamma=1e4): # 200 * 100 * 1
    s = t - t.transpose(-1, -2) # Pairwise time differences, shape [B, S, S]
    diag = torch.eye(t.shape[-2]).to(t) * 1e-5 # for numerical stability 100 * 100
    return torch.exp(-torch.square(s)*gp_gamma) + diag




def matern_kernel(r, sigma_f=1.0, l=1.0, nu=1.5):
    """
    Matern kernel function for Gaussian Process (PyTorch version).
    
    Parameters:
    - x, x_prime: Input tensors, the points for which to compute the kernel.
    - sigma_f: Signal variance (default is 1.0).
    - l: Length scale (default is 1.0).
    - nu: Smoothness parameter (default is 1.5).
    
    Returns:
    - The computed Matern kernel value between x and x_prime.
    """
    if nu == 0.5:
        # Exponential kernel (Matern with nu=0.5)
        return sigma_f**2 * torch.exp(-r / l)
    
    elif nu == 1.5:
        # Matern kernel with nu=1.5
        return sigma_f**2 * (1 + torch.sqrt(torch.tensor(3.0)) * r / l) * torch.exp(-torch.sqrt(torch.tensor(3.0)) * r / l)
    
    elif nu == 2.5:
        # Matern kernel with nu=2.5
        return sigma_f**2 * (1 + torch.sqrt(torch.tensor(5.0)) * r / l + 5 * r**2 / (3 * l**2)) * torch.exp(-torch.sqrt(torch.tensor(5.0)) * r / l)
    
    else:
        raise ValueError("Only nu values of 0.5, 1.5, and 2.5 are supported in this implementation.")





def plot_generate_git(basis, core, core_mean, core_std, data_std, data_mean, u_ind_uni, v_ind_uni, file_name = r"generate_animation.gif"):
    core = core * core_std + core_mean
    core = np.squeeze(core[0,:,:])

    output_batch = np.einsum("ik,lk->il", core, basis)

    output_batch = output_batch.reshape(output_batch.shape[0], u_ind_uni.shape[0], v_ind_uni.shape[0])
    data = output_batch * data_std + data_mean
    # 创建图形
    fig, ax = plt.subplots(figsize=(3, 2))
    n_frames = output_batch.shape[0]  # 帧数
    vmin = np.min(data[:n_frames,:,:])
    vmax = np.max(data[:n_frames,:,:])
    # 初始化图像，假设我们一开始绘制第一个矩阵
    im = ax.imshow(data[0], cmap='viridis', vmin=vmin, vmax=vmax)
    # 更新函数：用于在每一帧更新图像
    def update(frame):
        im.set_array(data[frame])  # 更新显示的矩阵
        #ax.set_title(f'Frame {frame+1}')  # 可以为每一帧设置标题
        return [im]

    # 创建动画：frames表示帧数，interval表示每帧之间的时间间隔（毫秒）
    ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)

    # 显示动画
    save_path = os.path.join("output", file_name)
    ani.save(save_path, writer='imagemagick', fps=20, dpi=100)




# compute rmse
def compute_rmse(gt, basis, core, core_mean, core_std, data_std, data_mean, u_ind_uni, v_ind_uni):
    core = core * core_std + core_mean
    core = np.squeeze(core[0,:,:])
    output_batch = np.einsum("ik,lk->il", core, basis)

    output_batch = output_batch.reshape(output_batch.shape[0], u_ind_uni.shape[0], v_ind_uni.shape[0])
    data = output_batch * data_std + data_mean
    return np.sqrt(np.mean((gt - data)**2)), np.mean(np.abs((gt - data)))





def load_meta_data(data_path):
    d = np.load(data_path, allow_pickle=True).item() # time, lat, lon, depth
    d = d["data"]
    u_uni = d["u_ind_uni"]
    v_uni = d["v_ind_uni"]
    w_uni = d["w_ind_uni"]
    t_uni = d["t_ind_uni"]

    return (u_uni, v_uni, w_uni, t_uni)


