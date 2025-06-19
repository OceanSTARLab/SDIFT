
import torch
from torch import nn
from typing import List, Callable
import math
dtype = torch.cuda.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega_0=1):
        super().__init__()
        self.omega_0 = omega_0
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.in_features,1 / self.in_features)
            #self.linear.weight.uniform_(-1 , 1 )

    def forward(self, input):
        return torch.sin(torch.sin(self.omega_0 * self.linear(input)))



class Continuous_Tucker_ssf(nn.Module):
    def __init__(self, r_1, r_2,  r_3, core):
        super(Continuous_Tucker_ssf, self).__init__()
        self.r_1 = r_1
        self.r_2 = r_2
        self.r_3 = r_3
    
        mid_channel = 512
        omega = 4
        self.U_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega),
                                   nn.Linear(mid_channel, r_1))

        self.V_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega),
                                   nn.Linear(mid_channel, r_2))

        self.W_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega),
                                   nn.Linear(mid_channel, r_3))



        self.core = core





    def kronecker_product_einsum_batched(self, A: torch.Tensor, B: torch.Tensor):
        """
        Batched Version of Kronecker Products
        :param A: has shape (b, a, c)
        :param B: has shape (b, k, p)
        :return: (b, ak, cp)
        """
        assert A.dim() == 3 and B.dim() == 3

        res = torch.einsum("bac,bkp->bakcp", A, B).view(A.size(0),
                                                        A.size(1) * B.size(1),
                                                        A.size(2) * B.size(2))
        return res

    def forward(self, train_ind_batch):


        U_input = train_ind_batch[:, 0].unsqueeze(1)
        V_input = train_ind_batch[:, 1].unsqueeze(1)
        W_input = train_ind_batch[:, 2].unsqueeze(1)




        U = self.U_net(U_input).unsqueeze(1)  # B * 1 * r_1
        V = self.V_net(V_input).unsqueeze(1)  # B * 1 * r_2
        W = self.W_net(W_input).unsqueeze(1) # B * 1 * r_3


        UV = self.kronecker_product_einsum_batched(U, V)
        UVW = self.kronecker_product_einsum_batched(UV, W).squeeze(1)

        out_put = torch.einsum("bi, i->b", UVW, self.core)
        return out_put







class Tensor_inr_3D(nn.Module):
    def __init__(self, R:tuple, omega=10):
        super(Tensor_inr_3D, self).__init__()
        self.r_1 = R[0]
        self.r_2 = R[1]
        self.r_3 = R[2]
        self._mode = "training"

        mid_channel = 1024
      
        self.U_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega), nn.Dropout(0),
                                   nn.Linear(mid_channel, self.r_1), nn.Tanh())

        self.V_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega), nn.Dropout(0),
                                   nn.Linear(mid_channel, self.r_2), nn.Tanh())
        
        self.W_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega), nn.Dropout(0),
                                   nn.Linear(mid_channel, self.r_3), nn.Tanh())
        



    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode not in ["training", "sampling"]:
            raise ValueError("Mode should be 'training' or 'sampling'")
        self._mode = mode


    def kronecker_product_einsum_batched(self, A: torch.Tensor, B: torch.Tensor):
        """
        Batched Version of Kronecker Products
        :param A: has shape (b, a, c)
        :param B: has shape (b, k, p)
        :return: (b, ak, cp)
        """
        assert A.dim() == 3 and B.dim() == 3

        res = torch.einsum("bac,bkp->bakcp", A, B).view(A.size(0),
                                                        A.size(1) * B.size(1),
                                                        A.size(2) * B.size(2))
        return res

    def forward(self, input_ind_train=None, input_ind_sampl=None):
        # input_ind_train: (U_ind_batch, V_ind_batch, W_ind_batch)
        # U_ind_batch: B * 1
        # V_ind_batch: B * 1
        # W_ind_batch: B * 1
        if self._mode == "training":
            U = self.U_net(input_ind_train[0].unsqueeze(1))  # B  * r_1
            V = self.V_net(input_ind_train[1].unsqueeze(1))  # B  * r_2
            W = self.W_net(input_ind_train[2].unsqueeze(1)) # B * r_3
            return (U,V,W)
        elif self._mode == "sampling":
        # input_ind_sampl: B * 3
            U = self.U_net(input_ind_sampl[:,:1]).unsqueeze(1)  # B * 1 * r_1
            V = self.V_net(input_ind_sampl[:,1:2]).unsqueeze(1)  # B * 1 * r_2
            W = self.W_net(input_ind_sampl[:,2:3]).unsqueeze(1) # B * 1 * r_3
            UV = self.kronecker_product_einsum_batched(U, V)
            UVW = self.kronecker_product_einsum_batched(UV, W).squeeze(1)
            return UVW

    


class Tensor_inr_4D(nn.Module):
    def __init__(self, R:tuple, omega=10):
        super(Tensor_inr_4D, self).__init__()
        self.r_1 = R[0]
        self.r_2 = R[1]
        self.r_3 = R[2]
        self.r_4 = R[3]
        self._mode = "training"

        mid_channel = 1024


        self.T_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega), nn.Dropout(0),
                                   nn.Linear(mid_channel, self.r_1), nn.Tanh())
      
        self.U_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega), nn.Dropout(0),
                                   nn.Linear(mid_channel, self.r_2), nn.Tanh())

        self.V_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega), nn.Dropout(0),
                                   nn.Linear(mid_channel, self.r_3), nn.Tanh())
        
        self.W_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega), nn.Dropout(0),
                                   nn.Linear(mid_channel, self.r_4), nn.Tanh())
        



    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode not in ["training", "sampling"]:
            raise ValueError("Mode should be 'training' or 'sampling'")
        self._mode = mode


    def kronecker_product_einsum_batched(self, A: torch.Tensor, B: torch.Tensor):
        """
        Batched Version of Kronecker Products
        :param A: has shape (b, a, c)
        :param B: has shape (b, k, p)
        :return: (b, ak, cp)
        """
        assert A.dim() == 3 and B.dim() == 3

        res = torch.einsum("bac,bkp->bakcp", A, B).view(A.size(0),
                                                        A.size(1) * B.size(1),
                                                        A.size(2) * B.size(2))
        return res

    def forward(self, input_ind_train=None):
        # input_ind_train: (U_ind_batch, V_ind_batch, W_ind_batch)
        # U_ind_batch: B * 1
        # V_ind_batch: B * 1
        # W_ind_batch: B * 1
        if self._mode == "training":
            T = self.T_net(input_ind_train[0].unsqueeze(1))
            U = self.U_net(input_ind_train[1].unsqueeze(1))  # B  * r_1
            V = self.V_net(input_ind_train[2].unsqueeze(1))  # B  * r_2
            W = self.W_net(input_ind_train[3].unsqueeze(1)) # B * r_3
            return (T,U,V,W)


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_value: float):
        super().__init__()
        self.max_value = max_value

        linear_dim = dim // 2
        periodic_dim = dim - linear_dim

        self.scale = torch.exp(-2 * torch.arange(0, periodic_dim).float() * math.log(self.max_value) / periodic_dim)
        self.shift = torch.zeros(periodic_dim)
        self.shift[::2] = 0.5 * math.pi

        self.linear_proj = nn.Linear(1, linear_dim)

    def forward(self, t):
        periodic = torch.sin(t * self.scale.to(t) + self.shift.to(t))
        linear = self.linear_proj(t / self.max_value)
        return torch.cat([linear, periodic], -1)

class FeedForward(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, activation: Callable=nn.ReLU(), final_activation: Callable=None):
        super().__init__()

        hidden_dims = hidden_dims[:]
        hidden_dims.append(out_dim)

        layers = [nn.Linear(in_dim, hidden_dims[0])]

        for i in range(len(hidden_dims) - 1):
            layers.append(activation)
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

        if final_activation is not None:
            layers.append(final_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



    
class TransformerModel(nn.Module): # This is the model we will train  
    def __init__(self, dim, hidden_dim, max_i, num_layers=8, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.t_enc = PositionalEncoding(hidden_dim, max_value=1)
        self.i_enc = PositionalEncoding(hidden_dim, max_value=max_i)

        self.input_proj = FeedForward(dim, [], hidden_dim)
        #self.input_proj = nn.Sequential(nn.Linear(dim, 2*hidden_dim), nn.ReLU(), nn.Linear(2*hidden_dim, 2*hidden_dim), nn.ReLU(), nn.Linear(2*hidden_dim, hidden_dim))

        self.proj = FeedForward(3 * hidden_dim, [], hidden_dim, final_activation=nn.ReLU())
        #self.proj = nn.Sequential(nn.Linear(3* hidden_dim, 8*hidden_dim), nn.ReLU(), nn.Linear(8*hidden_dim, 8*hidden_dim), nn.ReLU(), nn.Linear(8*hidden_dim, hidden_dim), nn.ReLU())

        self.enc_att = []
        self.i_proj = []
        self.linear = []
        self.linear2 = []
        for _ in range(num_layers):
            self.enc_att.append(nn.MultiheadAttention(hidden_dim, num_heads=2, batch_first=True))
            self.linear.append(nn.Sequential(nn.Linear(hidden_dim, 4*hidden_dim), nn.ReLU(), nn.Linear(4*hidden_dim, 4*hidden_dim), nn.ReLU(), nn.Linear(4*hidden_dim, hidden_dim)))
            self.linear2.append(nn.Sequential(nn.Linear(hidden_dim, 4*hidden_dim), nn.ReLU(), nn.Linear(4*hidden_dim, 4*hidden_dim), nn.ReLU(), nn.Linear(4*hidden_dim, hidden_dim)))
            self.i_proj.append(nn.Linear(3 * hidden_dim, hidden_dim))

        self.enc_att = nn.ModuleList(self.enc_att)
        self.i_proj = nn.ModuleList(self.i_proj)
        self.linear = nn.ModuleList(self.linear)
        self.linear2 = nn.ModuleList(self.linear2)

        #self.output_proj = FeedForward(hidden_dim, [], dim)
        self.output_proj = nn.Sequential(nn.Linear(hidden_dim, 8*hidden_dim), nn.ReLU(), nn.Linear(8*hidden_dim, 16*hidden_dim), nn.ReLU(),  nn.Linear(16*hidden_dim, 8*hidden_dim), nn.ReLU(), nn.Linear(8*hidden_dim, dim))

    def forward(self, x, t, i): #x:200 * 100 * 1, t:200 * 100 * 1, i:200 * 100 * 1
        shape = x.shape # shape = 200 * 100 * 1

        x = x.view(-1, *shape[-2:]) # x = 20000 * 100 * 1
        t = t.view(-1, shape[-2], 1)
        i = i.view(-1, shape[-2], 1)

        x = self.input_proj(x) # b t d
        t = self.t_enc(t)
        i = self.i_enc(i)

        x = self.proj(torch.cat([x, t, i], -1)) # time + index
        #x = self.proj2(x)

        # for att_layer, i_proj in zip(self.enc_att, self.i_proj):
        #     y, _ = att_layer(query=x, key=x, value=x)
        #     x = x + torch.relu(y)

        for att_layer,  l_layer1, l_layer2 in zip(self.enc_att, self.linear, self.linear2):
            y, _ = att_layer(query=x, key=x, value=x)
            x = x + torch.relu(l_layer1(y))
            #x = l_layer2(x)

        x = self.output_proj(x)
        x = 5*torch.tanh(x)
        x = x.view(*shape)
        return x
    

