import argparse
import os
join = os.path.join
import numpy as np
import torch
from tqdm import tqdm
import random
import scipy.io as sio
from networks_edm import *
import time


extensions = ['*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.bmp']

#----------------------------------------------------------------------------
# EDM sampler & EDM model

def set_seed(seed: int = 42):
    random.seed(seed)                    
    np.random.seed(seed)                  
    torch.manual_seed(seed)                
    torch.cuda.manual_seed(seed)           
    torch.cuda.manual_seed_all(seed)       

    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False     



def get_ktT(y_tt, core_t, gp_gamma=1, gp_sigma=1):
    r = torch.sqrt(torch.square(y_tt - core_t)) #  time differences, shape [B, S]
    return gp_sigma*torch.exp(-torch.square(r)*gp_gamma) # [B, S]



def get_kTT_inv(t, gp_gamma=1, gp_sigma=1):
    r = torch.sqrt(torch.square(t - t.transpose(-1, -2))) # Pairwise time differences, shape [B, S, S]
    diag = torch.eye(t.shape[-2]).to(t) * 1e-3 # for numerical stability 100 * 100
    K = gp_sigma*torch.exp(-torch.square(r)*gp_gamma) + diag
    #L = torch.linalg.cholesky(K.squeeze(0))
    #return (L.T@L).unsqueeze(0)
    return torch.inverse(K)
   
    




def compute_continuous_poest(x_0, basis_function, core_mean, core_std, core_t, y_group, ind_conti_group, y_time_group,  y_time_ind_group):
    ### input: x_0: estimated clean core 1 * T * R1 * R2 * R3; y_group:observation group;
    #          ind_conti_group: observation index group; t: time index of sampled core; post_t: time index of observation
    ### output: poest_matrix: continuous poestrior matrix
    core_tensor_shape = x_0.shape

    x_0 = x_0.view(core_tensor_shape[0], core_tensor_shape[1], -1) # vectorize [1, T, R1R2R3]
    poest_matrix1 = torch.zeros_like(x_0).to(device)  # Store the posterior gradients of stage1
    poest_matrix2 = torch.zeros_like(x_0).to(device)  # Store the posterior gradients of stage2
    x_0 = x_0 * core_std + core_mean #normalize 
    for y, y_tt, y_t_ind, ind in zip(y_group, y_time_group, y_time_ind_group, ind_conti_group): #A^{H}(y-Ax)
        if y.shape[0] == 0: # Skip if no observations are available for this time step
            continue
        #stage1: compute poesterior gradient for cores at the observation timestep
        x_0_t = x_0[0, y_t_ind, :] # [R1R2R3]
        y = torch.DoubleTensor(y).to(device)#[n]
        ind_conti = torch.FloatTensor(ind).to(device)#[n,3]
        A = basis_function(input_ind_sampl=ind_conti).detach() #[n,T]
        A = A.double()
        poest_matrix1[0, y_t_ind,:] = (A.T @ (y - A @ x_0_t)) # [R1R2R3]
        
        
        #stage2: compute continuos poesterior gradient for all cores 
        t_remove_group = y_time_ind_group.copy()
        t_remove_group.remove(y_t_ind)

        core_t_remove = core_t[:, t_remove_group, :] # [1,T-1]
        x_0_remove = x_0[:, t_remove_group, :] # [1,T-1,R1R2R3]


    

        ktT = get_ktT(y_tt, core_t_remove).squeeze(2) #[1,T-1]
        KTT_inv = get_kTT_inv(core_t_remove) #[T-1, T-1]
        #cov = (ktT @ KTT_inv @ ktT.T).to(device).squeeze()#[1,1]
        coeff = ((ktT @ KTT_inv).to(device)).squeeze(1)#[1,T]
        #coeff = coeff/(torch.abs(coeff).max())
        
        #cov = cov.squeeze() * A  @ A.T
        #cov_inv = cov.inverse() # [n,n]  
        x_0_aggregate = (coeff @ x_0_remove).squeeze()#[R1R2R3]
        post = (A.T @ (y - A @ x_0_aggregate)) # [R1R2R3]
        temp = torch.zeros_like(x_0).to(device)
        temp[:, t_remove_group,:] = torch.kron(coeff.unsqueeze(2), post.unsqueeze(0).unsqueeze(0)) #[1,T-1,R1R2R3]
        poest_matrix2 += temp

    return (poest_matrix1+MPDPS*poest_matrix2).view(core_tensor_shape) 


def wrapped_forward(x_hat, i_hat, t):
    return  edm(x_hat, i_hat, t,  use_ema=True).to(torch.float64)

@torch.no_grad()
def edm_post_sampler(
    edm, basis_function, latents, t, y_group, ind_conti_group, y_time_group,  y_time_ind_group, 
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    use_ema=True, zeta_i = 0.01):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, edm.sigma_min)
    sigma_max = min(sigma_max, edm.sigma_max)

    # Diffusion step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    i_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    i_steps = torch.cat([edm.round_sigma(i_steps), torch.zeros_like(i_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * i_steps[0]

    # cov matrix 
    KTT_inv =  get_kTT_inv(t)
    print("sampling")
    t_start = time.time()

    for i, (i_cur, i_next) in tqdm(enumerate(zip(i_steps[:-1], i_steps[1:]))): # 0, ..., N-1
        x_hat = x_next
        i_hat = i_cur
        
        # Euler step.
        denoised = edm(x_hat, i_hat, t,  use_ema=use_ema).to(torch.float64)
        denoised_core1 = denoised.detach().clone()
        d_cur = (x_hat - denoised) / i_hat
        x_next = x_hat + (i_next - i_hat) * d_cur
        
        x_next1 = x_next
        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = edm(x_next, i_next, t, use_ema=use_ema).to(torch.float64)
            denoised_core2 = denoised.detach().clone()
            #denoised = wrapped_forward(x_hat, i_hat, t)
            def wrapped_forward(x):
                return edm(x.unsqueeze(0), i_next, t, use_ema=use_ema).to(torch.float64)
            
            #J = vmap(jacrev(wrapped_forward))(x_next1)
  
            
            d_prime = (x_next - denoised) / i_next
            x_next = x_hat + (i_next - i_hat) * (0.5 * d_cur + 0.5 * d_prime)

            #Add llk_grad
            denoised_core = (denoised_core1 + denoised_core2) / 2
            llk_grad2 = compute_continuous_poest(denoised_core, basis_function, core_mean, core_std, t, y_group, ind_conti_group, y_time_group, y_time_ind_group)
            
            #x_next = x_next + (0.01/(i+1))*llk_grad2
            x_next = x_next + (zeta/(i+1))*llk_grad2
          
      



    t_end = time.time()
    print(f"Elapsed time: {t_end - t_start:.4f} seconds")
    print("sampling ",num_steps," steps completed")
    return x_next



def create_mask(shape, r):
    num_ones = int(np.prod(shape)*(r))
    mask = np.zeros(np.prod(shape), dtype=int)
    ones_indices = np.random.choice(np.prod(shape), num_ones, replace=False)
    mask[ones_indices] = 1
    mask = mask.reshape(shape)
    return mask

def get_te_observations(d, rho=0.02, mode=1, ind=0): # Randomly select observations from the test set
    d_shape = d.shape # [T, I1, I2, I3]
    data_extract = d[ind]
    mask = create_mask(data_extract.shape, rho)
    if mode == 1:
        pass
    elif mode == 2:
        mask[1::2, :, :, :] = 0 # Interval sampling
    

    # extract samples
    t_ind_uni = np.linspace(0, d_shape[1]-1, d_shape[1]).astype(int)/(d_shape[1]-1)
    if d_shape[2] == 1:
        u_ind_uni = np.ones_like([1])
    else:
        u_ind_uni = np.linspace(0, d_shape[2]-1, d_shape[2]).astype(int)/(d_shape[2]-1)
    v_ind_uni = np.linspace(0, d_shape[3]-1, d_shape[3]).astype(int)/(d_shape[3]-1)
    w_ind_uni = np.linspace(0, d_shape[4]-1, d_shape[4]).astype(int)/(d_shape[4]-1)

    indices_ob = np.where(mask == 1) # Extract the index of the observed data
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



def decoder(u_ind_uni, v_ind_uni,w_ind_uni, core, basis_function):
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", type=str, default="sampling", help="experiment name")
    parser.add_argument("--dataset", type=str, default="am")
    parser.add_argument('--seed', default=123, type=int, help='global seed')


    # EDM models parameters
    parser.add_argument('--gt_guide_type', default='l2', type=str, help='gt_guide_type loss type')
    parser.add_argument('--sigma_min', default=0.002, type=float, help='sigma_min')
    parser.add_argument('--sigma_max', default=80.0, type=float, help='sigma_max')
    parser.add_argument('--rho', default=7., type=float, help='Schedule hyper-parameter')
    parser.add_argument('--sigma_data', default=0.5, type=float, help='sigma_data used in EDM for c_skip and c_out')
    # Sampling parameters
    parser.add_argument('--total_steps', default=20, type=int, help='total_steps')
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--sample_mode", type=str, default='fid', help='sample mode')
    parser.add_argument('--begin_ckpt', default=0, type=int, help='begin_ckpt')

    # Model architecture (same as trained model)
    parser.add_argument("--img_size", type=int, default=48)
    parser.add_argument('--channels', default=1, type=int, help='input_output_channels')
    parser.add_argument('--model_channels', default=40, type=int, help='model_channels')
    parser.add_argument('--channel_mult', default=[1,2,2], type=int, nargs='+', help='channel_mult')
    parser.add_argument('--attn_resolutions', default=[], type=int, nargs='+', help='attn_resolutions')
    parser.add_argument('--num_layers', default=4, type=int,  help='num_layers')
    parser.add_argument('--layers_per_block', default=4, type=int, help='num_blocks')
    parser.add_argument('--num_temporal_latent', default=8, type=int, help='num_temporal_latent')
    config = parser.parse_args()

    from train_GPSD import EDM
    from train_GPSD import create_model
    from train_GPSD import get_gp_covariance



    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
   

    ## set random seed everywhere
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)  # for multi-GPU.
    random.seed(config.seed)  # Python random module.
    torch.manual_seed(config.seed)
    ## init model
    my_net = create_model(config)
    edm = EDM(model=my_net, cfg=config)

    # Free any unused GPU memory
    torch.cuda.empty_cache()

    # load core mean and variance 
    core_mean_std_path = r"./exps/gp-edm_am_12M/core_mean_std.mat"
    d = sio.loadmat(core_mean_std_path)
    core_mean = torch.tensor(d['core_mean'], dtype=torch.float32).to(device)
    core_std = torch.tensor(d['core_std'], dtype=torch.float32).to(device)


    # load baisis function
    basis_path = r"./ckp/basis4_am_2D_1x48x48_2025_05_04_14.pth" 
    basis_function = torch.load(basis_path, map_location=device)
    basis_function.eval()
    basis_function.mode = "sampling"



    # load generative model
    model_path = r"./exps/gp-edm_am_12M/checkpoints/ema_10000.pth" #
    ## load model
    checkpoint = torch.load(model_path, map_location=device)
    edm.model.load_state_dict(checkpoint)
    for param in edm.model.parameters():
        param.requires_grad = False
    edm.model.eval()

    rmse_list = []
    recon_list = []
    mask_list = []
    #hyperparameters
    rho = 0.01
    mt = 1
    MPDPS = 0.4
    zeta = 0.009 #0.01
    #zeta = 0.004 # 0.03
    # load observations
    te_data_path = r"./data/active_matter_te_20.npy"
    d = np.load(te_data_path, allow_pickle=True).item() # time, lat, lon, depth
    d = d["data"]
    #ind_list = [10]  # Specify  the index of the data to be sampled
    for i in tqdm(range(d.shape[0])):
    #for i in ind_list:
        
        data_extract, mask, y_group, ind_conti_group, y_time_group, y_time_ind_group, u_ind_uni, v_ind_uni, w_ind_uni = get_te_observations(d, rho=rho, mode=mt, ind=i)

        mask_list.append(mask)


        sample_shape= [1,d.shape[-4],1,48,48] # sampling core shape
        t_grid = (torch.linspace(0, 1, sample_shape[1]).view(1, -1, 1).to(device)).repeat(sample_shape[0], 1, 1)
        t_grid = t_grid.double()
        cov_sample = get_gp_covariance(t_grid)
        L_sample = torch.linalg.cholesky(cov_sample).to(device)
        noise_sample = torch.randn(sample_shape).to(device).double()
        
        x_T = (L_sample @ noise_sample.view(sample_shape[0], sample_shape[1],-1) ).view(sample_shape) # X_T
        sample = edm_post_sampler(edm, basis_function, x_T, t_grid, y_group, ind_conti_group, y_time_group, y_time_ind_group, num_steps=config.total_steps, use_ema=False).detach()
        core_sample = (sample*core_std + core_mean)
        out = decoder(u_ind_uni, v_ind_uni,w_ind_uni, core_sample[0], basis_function)
        recon_list.append(out)
        rmse = np.sqrt(np.mean((out - data_extract)**2))
        rmse_list.append(rmse)
        print("RMSE:", rmse)
        basis_function.mode = "sampling"

    recon_list = np.array(recon_list)
    mask_list = np.array(mask_list)
    rmse_list = np.array(rmse_list)
    rmse_mean = np.mean(rmse_list)
    rmse_std = np.std(rmse_list)
    print("mean:",rmse_mean, ";std:",rmse_std)
    sio.savemat(r'./results/'+config.dataset+'_mpdps_'+str(MPDPS)+'_recon_rho'+str(rho)+'_mode_'+str(mt)+'mean_'+str(rmse_mean)+'_std_'+str(rmse_std)+'.mat', {"recon_list": recon_list, "mask_list":mask_list, "rmse_list":rmse_list}) # saving reconstructions results


        