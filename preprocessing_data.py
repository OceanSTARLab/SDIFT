import numpy as np
import h5py
from tqdm import tqdm




        



class PDEDataPreprocesser_3D_large_data(): 
    def __init__(self, data_path, data_name, tr_num):
        with h5py.File(data_path, 'r') as f:
            data = f['data'][:170]
            
        print("read data completed")
        self.full_data = data
        self.shape = self.full_data.shape # B*T*H*W*D
        self.name = data_name
        self.tr_num = tr_num
        self.indices_tr = np.random.permutation(self.shape[0])[:self.tr_num]
        self.indices_te = np.random.permutation(self.shape[0])[self.tr_num:]


    def create_mask(self, shape, r):
        num_ones = int(np.prod(shape)*(r))
        mask = np.zeros(np.prod(shape), dtype=int)
        ones_indices = np.random.choice(np.prod(shape), num_ones, replace=False)
        mask[ones_indices] = 1
        mask = mask.reshape(shape)
        return mask


    def pde_preprocessing_3D_bench(self): # Preprocess 3D data and generate training，testing data and metadata.
        # create observation mask
        print("shape:",self.shape)
        t_ind_uni = np.linspace(0, self.shape[1]-1, self.shape[1]).astype(int)/(self.shape[1]-1)
        if self.shape[2] == 1:
            u_ind_uni = np.ones_like([1])
        else:
            u_ind_uni = np.linspace(0, self.shape[2]-1, self.shape[2]).astype(int)/(self.shape[2]-1)
        v_ind_uni = np.linspace(0, self.shape[3]-1, self.shape[3]).astype(int)/(self.shape[3]-1)
        w_ind_uni = np.linspace(0, self.shape[4]-1, self.shape[4]).astype(int)/(self.shape[4]-1)

        data_extract = self.full_data


        mask_tr = self.create_mask(self.shape[1:], r=0.1)

        

        data_dict = dict({})
        data_dict["u_ind_uni"] = u_ind_uni
        data_dict["v_ind_uni"] = v_ind_uni
        data_dict["w_ind_uni"] = w_ind_uni
        data_dict["t_ind_uni"] = t_ind_uni
        data_dict["mask_tr"] = mask_tr

        data_norm = data_extract[self.indices_tr] 
        record_size = self.shape[1:]
            
        num_samples = self.tr_num
        chunk_size = int(self.tr_num/10)
        print("chunk_size:", chunk_size)

        with h5py.File(r'./data/'+str(self.name)+'_tr_data_'+str(self.tr_num)+'.h5', 'w') as f: 
            # Create a training dataset (pre-defined size), using chunk and gzip compression. (for large training data)
            dset = f.create_dataset(
                'data',
                shape=(num_samples, *record_size),
                dtype='float32',
                chunks=(chunk_size, *record_size),      
                compression='gzip'         
            )

            # Write data in batches
            for i in tqdm(range(0, num_samples, chunk_size)):
                data = data_norm[i:i+chunk_size]
                dset[i:i+chunk_size] = data

        np.save(r"./data/"+self.name+"_te_"+str(self.shape[0]-self.tr_num)+".npy", {"data" : data_extract[self.indices_te]})
        np.save(r"./data/"+self.name+"_tr_metadata_"+str(self.tr_num)+".npy", {"data" : data_dict})
        




class PDEDataProcess_inference_3D_large_data(): # 预处理采样推断3D数据
    def __init__(self, data_path, data_name):
        d = np.load(data_path, allow_pickle=True).item()
        self.full_data = d["data"]
        self.shape = self.full_data.shape # B*T*H*W*D
        self.name = data_name
     

    def create_mask(self, shape, r):
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


    # 测试数据采样需要重新写，从te文件中读取
    def get_pde_test_3D(self, rho, mt, ind):
        # create observation mask
        

        data_extract = self.full_data
        data_extract = data_extract[ind,:,:,:,:] #提取第ind个数据
        print("shape:",data_extract.shape)

        if mt == "1":
            mask_te = self.create_mask(data_extract.shape, rho) #创建观测mask
        elif mt == "2":
            mask_te = self.create_mask(data_extract.shape, rho) #创建观测mask
            mask_te[::2, :, :, :] = 0 #每隔一个时刻抽样
        elif mt == "3":
            mask_te = self.create_mask(data_extract.shape, rho) #创建观测mask
            mask_te[::2, :, :, :] = 0 #每隔一个时刻抽样
            mask_te[14:, :, :, :] = 0 #后面的时刻不抽样
        

        data_dict = dict({})

        data_dict["data"] = data_extract
        data_dict["mask_te"] = mask_te

        np.save(r"./data/posterior_sampling/"+self.name+"_inference_rho_"+str(rho)+"_missingtype"+mt+"_"+str(ind)+".npy", {"data" : data_dict})



    def convert_observation(self, rho, mt,ind): # 将观测数据转换为dps所需格式
        data_path = r"./data/posterior_sampling/"+self.name+"_inference_rho_"+str(rho)+"_missingtype"+mt+"_"+str(ind)+".npy"
        d = np.load(data_path, allow_pickle=True).item() # time, lat, lon, depth
        d = d["data"]
        data_extract = d["data"]
        mask_ob = d["mask_te"]

        t_ind_uni = np.linspace(0, self.shape[1]-1, self.shape[1]).astype(int)/(self.shape[1]-1)
        if self.shape[2] == 1:
            u_ind_uni = np.ones_like([1])
        else:
            u_ind_uni = np.linspace(0, self.shape[2]-1, self.shape[2]).astype(int)/(self.shape[2]-1)
        v_ind_uni = np.linspace(0, self.shape[3]-1, self.shape[3]).astype(int)/(self.shape[3]-1)
        w_ind_uni = np.linspace(0, self.shape[4]-1, self.shape[4]).astype(int)/(self.shape[4]-1)

        indices_ob = np.where(mask_ob == 1) #提取观测数据的索引
        ob_ind_conti = np.array([t_ind_uni[indices_ob[0]], u_ind_uni[indices_ob[1]], v_ind_uni[indices_ob[2]], w_ind_uni[indices_ob[3]]]).T
        ob_ind = np.array(indices_ob).T
        ob_y = data_extract[indices_ob]


        data_dict = dict({})
        data_dict["u_ind_uni"] = u_ind_uni
        data_dict["v_ind_uni"] = v_ind_uni
        data_dict["w_ind_uni"] = w_ind_uni
        data_dict["t_ind_uni"] = t_ind_uni

        data_dict["ob_ind_conti"] = ob_ind_conti
        data_dict["ob_ind"] = ob_ind
        data_dict["ob_y"] = ob_y

   

        np.save(r"./data/posterior_sampling/"+self.name+"_mmps_rho_"+str(rho)+"_missingtype"+mt+"_"+str(ind)+".npy", {"data" : data_dict})


        



if __name__ == "__main__":
    ####******************************************************************************
    data_name = r"active_matter"
    data_path = r"./data/active_matter_928_2.h5" # raw_data 
    d = PDEDataPreprocesser_3D_large_data(data_path, data_name, tr_num=150)
    d.pde_preprocessing_3D_bench() # training dataset

    # # active matter inference data
    # inference_data_path = r"./data/active_matter_te_44.npy" # raw_data
    # rho = 0.01
    # missing_type = "3"
    # d = PDEDataProcess_inference_3D_large_data(inference_data_path, data_name, ind=0)
    # d.get_pde_test_3D(rho, missing_type) # test dataset
    # d.convert_observation(rho, missing_type) # observation for dps
    # print("done")   
    # ******************************************************************************


    # #******************************************************************************
    # data_name = r"SSF"
    # data_path = r"./data/ssf_data_1000.h5" # raw_data 
    # d = PDEDataPreprocesser_3D_large_data(data_path, data_name, tr_num=150)
    # d.pde_preprocessing_3D_bench() # training dataset

    # # active matter inference data
    # inference_data_path = r"./data/SSF_te_50.npy" # raw_data
    # rho = 0.02
    # missing_type = "1"
    # ind = 1
    # d = PDEDataProcess_inference_3D_large_data(inference_data_path, data_name)
    # d.get_pde_test_3D(rho, missing_type, ind) # test dataset
    # d.convert_observation(rho, missing_type, ind) # observation for dps
    # print("done")   
    # #******************************************************************************





    
    # #******************************************************************************
    # data_name = r"supernova_explosion"
    # data_path = r"./data/supernova_explosion_396.h5" # raw_data 
    # d = PDEDataPreprocesser_3D_large_data(data_path, data_name, tr_num=150)
    # d.pde_preprocessing_3D_bench() # training dataset

    # # # active matter inference data
    # # inference_data_path = r"./data/active_matter_te_data_1000.npy" # raw_data
    # # rho = 0.01
    # # missing_type = "3"
    # # d = PDEDataProcess_inference_3D_large_data(inference_data_path, data_name, ind=0)
    # # d.get_pde_test_3D(rho, missing_type) # test dataset
    # # d.convert_observation(rho, missing_type) # observation for dps
    # # print("done")   
    # #******************************************************************************

