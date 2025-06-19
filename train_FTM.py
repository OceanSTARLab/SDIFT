#scaling the computation of tucker core for large-scale data
import numpy as np
import torch
from torch import  optim
from tqdm import tqdm
from FTM_model import Tensor_inr_3D
from torch.utils.data import TensorDataset, DataLoader
import scipy.io as scio
from datetime import datetime
from utils import *
import argparse


current_time = datetime.now()


def loss_fn(pred, gt, mask=None): #RMSE loss
    
    assert pred.size() == gt.size()
    diff = pred - gt
    if mask is not None:
        assert mask.size() == gt.size()
        diff = diff[mask==1]
    mse = torch.mean(diff ** 2)
    return torch.sqrt(mse)


def loss_fn2(pred, gt, mask=None): #MAE loss
    assert pred.size() == gt.size()
    diff = torch.abs(pred - gt)
    if mask is not None:
        assert mask.size() == gt.size()
        diff = diff[mask==1]
    return torch.mean(diff)


def training(basis_function, tucker_core,  train_loader, optimizer, loss_fn, loss_fn2):
    # set model to training mode
    basis_function.train()
    basis_function.mode = "training"
    # Use tqdm for progress bar
    loss_list = []
    for i, (data,  batch_ind) in enumerate(train_loader):

        mask_tmp = mask_tr.unsqueeze(0)
        mask_tmp = mask_tr.repeat(data.shape[0], 1, 1, 1, 1)
        mask_tmp = mask_tmp.to(device)
        optimizer.zero_grad()


        basises  = basis_function(input_ind_train = ind_input)  # (I1*R1, I2*R2, I3*R3)
        output = torch.einsum("mi, btijk->btmjk", basises[0], tucker_core[batch_ind,:,:,:,:])
        output = torch.einsum("nj, btmjk->btmnk", basises[1], output)
        output = torch.einsum("ok, btmnk->btmno", basises[2], output)
        #output = torch.tanh(f*output)

        loss = loss_fn(output, data, mask=mask_tmp) + total_variation_loss(tucker_core[batch_ind,:,:,:,:], weight=1e-7)


        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

    loss_mean = np.mean(loss_list)
    return loss_mean



def evaluating(basis_function, tucker_core, test_loader, loss_fn, loss_fn2):
    # set model to training mode
    basis_function.eval()
    # Use tqdm for progress bar

    print("evaluating....")

    for i, (data,   batch_ind) in enumerate(test_loader):

        basises  = basis_function(input_ind_train = ind_input)  # (I1*R1, I2*R2, I3*R3)
        #core = torch.sin(f*core)
        output = torch.einsum("mi, btijk->btmjk", basises[0], tucker_core[batch_ind,:,:,:,:])
        output = torch.einsum("nj, btmjk->btmnk", basises[1], output)
        output = torch.einsum("ok, btmnk->btmno", basises[2], output)
        rmse = loss_fn(output, data)
        mae = loss_fn2(output, data)
        del output
    
    return rmse.item(), mae.item()





def train_parallel(basis_function, tucker_core, train_loader, test_loader,  optimizer, loss_fn, loss_fn2, max_iter):
        loss_min = 10
        Epoch_list = []
        RMSE_list = []
        MAE_list = []
        RMSE_min = 10
        print("(rmse,mae):", evaluating(basis_function, tucker_core, test_loader, loss_fn, loss_fn2))
        for iter in tqdm(range(max_iter)):
            basis_function.mode = "training"
            loss = training(basis_function, tucker_core, train_loader,  optimizer, loss_fn, loss_fn2)
            if iter % 20 == 0:
                print("epoch:", iter, "RMSE loss:", loss, "\n")
            if iter > 800 and (loss< loss_min or iter%50==0):
                loss_min = loss
                rmse, mae = evaluating(basis_function, tucker_core, test_loader, loss_fn, loss_fn2)
                Epoch_list.append(iter)
                RMSE_list.append(rmse)
                MAE_list.append(mae)

                if rmse < RMSE_min:
                    RMSE_min = rmse
                    formatted_time = current_time.strftime("%Y_%m_%d_%H") 
                    scio.savemat("./data/core_"+config.data_name+"_"+str(R[0])+"x"+str(R[1])+"x"+str(R[2])+"_"+str(formatted_time)+".mat", {"core": tucker_core.detach().cpu().numpy()})
                    torch.save(basis_function, "./ckp/basis_"+config.data_name+"_"+str(R[0])+"x"+str(R[1])+"x"+str(R[2])+"_"+str(formatted_time)+".pth")
                    print("save core successfully")

                print("iter:", iter, ";evaluating RMSE = ", rmse, ";evaluating MAE = ", mae)





if __name__ == "__main__":
    if os.name == 'nt':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    print("device:", device)


    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="am_2D")
    parser.add_argument("--batch_size", type=int, default=512)   
    parser.add_argument("--data_path", type=str, default=r"./data/active_matter_tr_data_900_2.h5") #
    parser.add_argument("--metadata_path", type=str, default=r"./data/active_matter_tr_metadata_900_2.npy")
    parser.add_argument("--R", type=int, default=(1,48,48), help="size of Tucker core") 
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_iter", type=int, default=1000)
    config = parser.parse_args()

    ##########################data preprocessing#############################
    ind_uni, data_extract, mask_tr = load_large_data(config.data_path, config.metadata_path)
    print("load data completed")
    print("data_size:", data_extract.shape)
    print("mask_ratio:", np.sum(mask_tr)/mask_tr.size)



    u_ind_uni = torch.FloatTensor(ind_uni[0]).to(device)
    v_ind_uni = torch.FloatTensor(ind_uni[1]).to(device)
    w_ind_uni = torch.FloatTensor(ind_uni[2]).to(device)
    t_ind_uni = torch.FloatTensor(ind_uni[3]).to(device)

    ind_input = (u_ind_uni, v_ind_uni, w_ind_uni)

    

    data = torch.FloatTensor(data_extract).to(device)
    batch_ind = torch.arange(data.size()[0]).to(device)
    mask_tr = torch.FloatTensor(mask_tr).to(device)
   


    R = config.R
    print("data_name:",config.data_name,"R_size:", R)
    data_size = data.size()

 

    #set_random_seed(231)
    learning_rate = config.learning_rate    
    tucker_core = (torch.ones(data_size[0], data_size[1], R[0], R[1], R[2])/2).to(device) #BXTXR1XR2XR3  
    basis_function = Tensor_inr_3D(R, omega=20).to(device)



    params = []
    params += [x for x in basis_function.parameters()]
    tucker_core.requires_grad = True
    params += [tucker_core]
    optimizer = optim.AdamW(params, learning_rate)

    train_dataset = TensorDataset(data, batch_ind)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataset = TensorDataset(data,   batch_ind)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    train_parallel(basis_function, tucker_core, train_loader, test_loader, optimizer, loss_fn, loss_fn2, max_iter=config.max_iter)
    


