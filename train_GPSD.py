import argparse
import os
join = os.path.join
import torch
from tqdm import tqdm
import copy
import random
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.io as sio
from networks_edm import  Spatial_temporal_UNet
import scipy.io as scio
extensions = ['*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.bmp']

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2). (independent design) active matter
## https://github.com/NVlabs/edm/blob/main/generate.py#L25
## deterministic case
@torch.no_grad()
def edm_sampler(
    edm, latents, t,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    use_ema=True,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, edm.sigma_min)
    sigma_max = min(sigma_max, edm.sigma_max)

    # Diffusion step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    i_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    i_steps = torch.cat([edm.round_sigma(i_steps), torch.zeros_like(i_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * i_steps[0]
    print("sampling")
    for i, (i_cur, i_next) in enumerate(zip(i_steps[:-1], i_steps[1:])): # 0, ..., N-1
        x_hat = x_next
        i_hat = i_cur
        
        # Euler step.
        denoised = edm(x_hat, i_hat, t,  use_ema=use_ema).to(torch.float64)
        d_cur = (x_hat - denoised) / i_hat
        x_next = x_hat + (i_next - i_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = edm(x_next, i_next, t, use_ema=use_ema).to(torch.float64)
            d_prime = (x_next - denoised) / i_next
            x_next = x_hat + (i_next - i_hat) * (0.5 * d_cur + 0.5 * d_prime)
    print("sampling ",num_steps," steps completed")
    return x_next

def get_gp_covariance(t): # 200 * 100 * 1
    gp_gamma = 50
    s = t - t.transpose(-1, -2) # Pairwise time differences, shape [B, S, S]
    diag = torch.eye(t.shape[-2]).to(t) * 1e-5 # for numerical stability 100 * 100
    return torch.exp(-torch.square(s)*gp_gamma) + diag

#----------------------------------------------------------------------------
# EDM model

class EDM():
    def __init__(self, model=None, cfg=None):
        self.cfg = cfg
        self.device = self.cfg.device
        self.model = model.to(self.device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        ## parameters
        self.sigma_min = cfg.sigma_min
        self.sigma_max = cfg.sigma_max
        self.rho = cfg.rho
        self.sigma_data = cfg.sigma_data
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        self.ema_rampup_ratio = 0.05
        self.ema_halflife_kimg = 500

    def model_forward_wrapper(self, x, sigma, t, use_ema=False, **kwargs):
        """Wrapper for the model call"""
        sigma[sigma == 0] = self.sigma_min
        ## edm preconditioning for input and output
        ## https://github.com/NVlabs/edm/blob/main/training/networks.py#L632
        ### x: [B, T, C], sigma: [B], t: [B, T, 1]
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
  
        if use_ema:
            model_output = self.ema(torch.einsum('b,btijk->btijk', c_in, x), c_noise.view(-1, 1, 1).repeat(1, t.shape[1],  1), t)
        else:
            model_output = self.model(torch.einsum('b,btijk->btijk', c_in, x), c_noise.view(-1, 1, 1).repeat(1, t.shape[1],  1), t)
        # try:
        #     model_output = model_output.sample
        # except:
        #     pass
        return torch.einsum('b,btijk->btijk', c_skip, x) + torch.einsum('b,btijk->btijk', c_out, model_output)
        
    def train_step(self, signals, t,  **kwargs):
        ### sigma sampling --> continuous & weighted sigma
        ## https://github.com/NVlabs/edm/blob/main/training/loss.py#L66
        rnd_normal = torch.randn([signals.shape[0]], device=signals.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = signals
        cov = get_gp_covariance(t) # Covariance matrix 200*100*100
        L = torch.linalg.cholesky(cov) # 200*100*100

        noise = torch.randn_like(y)
        noise = L@(noise.view(y.shape[0], y.shape[1], -1))
        noise = noise.view(y.shape)

        n = torch.einsum('b,btijk->btijk', sigma, noise) # noise with sigma (scaling factor)
        D_yn = self.model_forward_wrapper(y + n, sigma, t)
        if self.cfg.gt_guide_type == 'l2':
            loss = torch.einsum('b,btijk->btijk', weight, ((D_yn - y) ** 2))
        elif self.cfg.gt_guide_type == 'l1':
            loss = torch.einsum('b,btijk->btijk', weight, (torch.abs(D_yn - y)))
        else:
            raise NotImplementedError(f'gt_guide_type {self.cfg.gt_guide_type} not implemented')
        return loss.mean()
    
    def update_ema(self):
        ema_halflife_nimg = self.ema_halflife_kimg * 1000
        if self.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, step * config.train_batch_size * self.ema_rampup_ratio)
        ema_beta = 0.5 ** (config.train_batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(self.ema.parameters(), self.model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
    
    # used for sampling, set use_ema=True
    def __call__(self, x, sigma, t, use_ema=True):
        if sigma.shape == torch.Size([]):
            sigma = sigma * torch.ones([x.shape[0]]).to(x.device)
        return self.model_forward_wrapper(x.float(), sigma.float(), t.float(), use_ema=use_ema)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

## mynet model creater
def create_model(config):
    unet = Spatial_temporal_UNet(in_channels=config.channels, 
                    out_channels=config.channels, 
                    num_blocks=config.layers_per_block, 
                    num_temporal_latent = config.num_temporal_latent,
                    attn_resolutions=config.attn_resolutions, 
                    model_channels=config.model_channels, 
                    channel_mult=config.channel_mult, 
                    dropout=0, 
                    img_resolution=config.img_size, 
                    label_dim=0,
                    embedding_type='positional', 
                    encoder_type='standard', 
                    decoder_type='standard', 
                    augment_dim=9, 
                    channel_mult_noise=1, 
                    resample_filter=[1,1], 
                    )
    pytorch_total_grad_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logging.info(f'total number of trainable parameters in the Score Model: {pytorch_total_grad_params}')
    pytorch_total_params = sum(p.numel() for p in unet.parameters())
    logging.info(f'total number of parameters in the Score Model: {pytorch_total_params}')
    return unet

def save_plots(samples, t_grid, sample_dir, step):
    for i in range(samples.shape[0]):
        plt.plot(t_grid[i].squeeze().detach().cpu().numpy(), samples[i, :, 0,0,0].squeeze().detach().cpu().numpy(), color='C0', alpha=1 / (i + 1))
    plt.title('10 new realizations')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.savefig(f'{sample_dir}/samples_{step}.png')
    plt.close()




def normalize_data(tr_y): # normalize data to [0,1]
    data_mean = tr_y.min()
    data_std = tr_y.max() - tr_y.min()
    tr = (tr_y - data_mean) / data_std
    return tr,  data_mean, data_std

def load_train_data(config):
    if config.dataset == 'am' or "ssf":
        ### create dataloader
        d = sio.loadmat(config.core_path)
        core = torch.tensor(d['core'], dtype=torch.float32).to(device)
        core, core_mean, core_std = normalize_data(core)
        #core = core[:50,...]
        print("core shape: ", core.shape,"core_mean:", core_mean,"core_std: ", core_std)
        t = (torch.linspace(0, 1, core.shape[1]).view(1, -1, 1).to(device)).repeat(core.shape[0], 1, 1) #core.shape[0], core.shape[1], 1

        data_set = torch.utils.data.TensorDataset(core, t)
        train_loader = torch.utils.data.DataLoader(data_set,
                                                    batch_size=config.train_batch_size,
                                                    shuffle=True,
                                                    num_workers=0)
        logger.info(f'length of training loader: {len(train_loader)}')
        return train_loader,  core_mean, core_std

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", type=str, default="gp-edm")
    parser.add_argument("--dataset", type=str, default="am")
    parser.add_argument("--core_path", type=str, default="./data/core3_am_2D_1x48x48_2025_04_28_16.mat")
    parser.add_argument('--seed', default=231, type=int, help='global seed')
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=15001)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--img_size", type=int, default=48)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--save_model_iters", type=int, default=10000)
    parser.add_argument("--log_step", type=int, default=500)
    parser.add_argument("--train_dataset", action='store_true', default=True)
    parser.add_argument("--desired_class", type=str, default='all')
    parser.add_argument("--train_progress_bar", action='store_true', default=True)
    parser.add_argument("--warmup", type=int, default=5000)
    # EDM models parameters
    parser.add_argument('--gt_guide_type', default='l2', type=str, help='gt_guide_type loss type')
    parser.add_argument('--sigma_min', default=0.002, type=float, help='sigma_min')
    parser.add_argument('--sigma_max', default=80.0, type=float, help='sigma_max')
    parser.add_argument('--rho', default=7., type=float, help='Schedule hyper-parameter')
    parser.add_argument('--sigma_data', default=0.5, type=float, help='sigma_data used in EDM for c_skip and c_out')
    # Sampling parameters
    parser.add_argument('--total_steps', default=20, type=int, help='total_steps')
    parser.add_argument("--save_signals_step", type=int, default=500)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument('--begin_ckpt', default=0, type=int, help='begin_ckpt')
    # Model architecture
    parser.add_argument("--img_size", type=int, default=48)
    parser.add_argument('--channels', default=1, type=int, help='input_output_channels')
    parser.add_argument('--model_channels', default=40, type=int, help='model_channels')
    parser.add_argument('--channel_mult', default=[1,2,2], type=int, nargs='+', help='channel_mult')
    parser.add_argument('--attn_resolutions', default=[], type=int, nargs='+', help='attn_resolutions')
    parser.add_argument('--num_layers', default=4, type=int,  help='number of layers in each block')
    parser.add_argument('--layers_per_block', default=4, type=int, help='num_blocks')
    parser.add_argument('--num_temporal_latent', default=8, type=int, help='num_temporal_latent')
    
    config = parser.parse_args()


    if os.name == 'nt':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config.device = device
    print("Device:",device)

    # workdir setup
    config.expr = f"{config.expr}_{config.dataset}"
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    outdir = f"exps/{config.expr}_{run_id}"
    os.makedirs(outdir, exist_ok=True)
    sample_dir = f"{outdir}/samples"
    os.makedirs(sample_dir, exist_ok=True)
    ckpt_dir = f"{outdir}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    logging.basicConfig(filename=f'{outdir}/std.log', filemode='w', 
                        format='%(asctime)s %(levelname)s --> %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.info("#################### Arguments: ####################")
    for arg in vars(config):
        logger.info(f"\t{arg}: {getattr(config, arg)}")
    print("-------------------------------------------------------------------------")
    print("outdir:", outdir)    

    ## set random seed everywhere
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)  # for multi-GPU.
    random.seed(config.seed)  # Python random module.
    torch.manual_seed(config.seed)

    ## load dataset
    train_loader, core_mean, core_std  = load_train_data(config)
    scio.savemat(f'{outdir}/core_mean_std.mat', {"core_mean": core_mean.cpu().numpy(), "core_std": core_std.cpu().numpy()})


    ## init model
    mynet = create_model(config)
    edm = EDM(model=mynet, cfg=config)
    edm.model.train()
    logger.info("#################### Model: ####################")
    # logger.info(f'{mynet}')
    num_para = sum(p.numel() for p in mynet.parameters() if p.requires_grad)
    logger.info(f'number of trainable parameters of phi model in optimizer: {num_para}')
    print("number of trainable parameters of phi model in optimizer: ", num_para)
    ## setup optimizer
    # optimizer = torch.optim.AdamW(edm.model.parameters(),lr=config.learning_rate)
    optimizer = torch.optim.Adam(edm.model.parameters(),lr=config.learning_rate)

    logger.info("#################### Training ####################")
    train_loss_values = 0
 

    if config.train_progress_bar:
        progress_bar = tqdm(total=config.num_steps)
    for step in range(config.num_steps):
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        # accumulation steps
        for _ in range(config.accumulation_steps):
            try:
                signal_batch, t_batch = next(data_iterator)
            except:
                data_iterator = iter(train_loader)
                signal_batch, t_batch = next(data_iterator)
    
            loss = edm.train_step(signal_batch, t_batch)
            loss /= (config.accumulation_steps)
            loss.backward()
            batch_loss += loss
        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = config.learning_rate * min(step / config.warmup, 1)
        for param in mynet.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()
        train_loss_values += (batch_loss.detach().item())
        ## Update EMA.
        edm.update_ema()
        ## Update state
        if config.train_progress_bar:
            logs = {"loss": loss.detach().item()}
            progress_bar.update(1) 
            progress_bar.set_postfix(**logs)
        ## log
        if step % config.log_step == 0 or step == config.num_steps - 1:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'step: {step:08d}, current lr: {current_lr:0.6f} average loss: {train_loss_values/(step+1):0.10f}; batch loss: {batch_loss.detach().item():0.10f}')
        ## save signals
        if config.save_signals_step and (step % config.save_signals_step == 0 or step == config.num_steps - 1):
            # generate data with the model to later visualize the learning process
            edm.model.eval()
            sample_shape= [5,24,config.channels,config.img_size,config.img_size]
            t_grid = (torch.linspace(0, 1, sample_shape[1]).view(1, -1, 1).to(device)).repeat(sample_shape[0], 1, 1)
            cov_sample = get_gp_covariance(t_grid)
            L_sample = torch.linalg.cholesky(cov_sample).to(device)
            noise_sample = torch.randn(sample_shape).to(device)
            
            x_T = (L_sample @ noise_sample.view(sample_shape[0], sample_shape[1],-1) ).view(sample_shape) # X_T
            sample = edm_sampler(edm, x_T, t_grid, num_steps=config.total_steps).detach()
            sample = (sample*core_std + core_mean).cpu()
            if step >=config.save_model_iters:
                sio.savemat(f'{sample_dir}/core_{str(step)}.mat', {"core": sample})
            save_plots(sample, t_grid, sample_dir, step)

            
        
        ## save model
        if config.save_model_iters and (step % config.save_model_iters == 0 or step == config.num_steps - 1) and step > 0:
            torch.save(edm.model.state_dict(), f"{ckpt_dir}/ema_{step}.pth")
            
        edm.model.train()
        