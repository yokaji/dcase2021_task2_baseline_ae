########################################################################
# import default libraries
########################################################################
import os
import sys
import gc
import random
########################################################################


########################################################################
# import additional libraries
########################################################################
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
# from import
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from torch import optim
try:
    from sklearn.externals import joblib
except:
    import joblib

# original lib
import common as com
from pytorch_model import AutoEncoder
########################################################################

########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################


########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(7, 5))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        self.plt.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)


########################################################################

########################################################################
# create Dataset class
########################################################################
class DCASE_Dataset(Dataset):
    def __init__(self, path_list, params):
        
        self.path_list = path_list
        self.params = params
        
        # culculate the number of dimentions
        dims = params["feature"]["n_mels"] * params["feature"]["n_frames"]
        
        # iterate to file_to_vector_array()
        for idx in tqdm(range(len(path_list))):
            vectors = com.file_to_vectors(path_list[idx],
                                                    n_mels=params["feature"]["n_mels"],
                                                    n_frames=params["feature"]["n_frames"],
                                                    n_fft=params["feature"]["n_fft"],
                                                    hop_length=params["feature"]["hop_length"],
                                                    power=params["feature"]["power"])
            vectors = vectors[: : params["feature"]["n_hop_frames"], :]
            if idx == 0:
                data = np.zeros((len(path_list) * vectors.shape[0], dims), np.float32)
            data[vectors.shape[0] * idx : vectors.shape[0] * (idx + 1), :] = vectors

        self.feat_data = data
        
    def __len__(self):
        return self.feat_data.shape[0]
    
    def __getitem__(self, idx):
        output = torch.tensor(self.feat_data[idx, :])
        return output

########################################################################
# main 00_train.py
########################################################################
if __name__ == "__main__":
    
    ####################################################################
    # set seed
    ####################################################################
    random.seed(57)
    np.random.seed(57)
    torch.manual_seed(57)
    torch.cuda.manual_seed(57)
    ####################################################################
    
    ####################################################################
    # set device
    ####################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device : {}".format(device))
    ####################################################################
    
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)
        
    # make output directory
    os.makedirs(param["model_directory"], exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # load base_directory list
    dirs = com.select_dirs(param=param, mode=mode)

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {target_dir}".format(target_dir=target_dir, idx=idx+1, total=len(dirs)))

        # set path
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"],
                                                                     machine_type=machine_type)

        if os.path.exists(model_file_path):
            com.logger.info("model exists")
            continue
        
        history_img = "{model}/history_{machine_type}.png".format(model=param["model_directory"],
                                                                  machine_type=machine_type)
        # pickle file for storing anomaly score distribution
        score_distr_file_path = "{model}/score_distr_{machine_type}.pkl".format(model=param["model_directory"],
                                                                                machine_type=machine_type)

        # generate dataset
        print("============== DATASET_GENERATOR ==============")

        # get file list for all sections
        # all values of y_true are zero in training
        files, y_true = com.file_list_generator(target_dir=target_dir,
                                                section_name="*",
                                                dir_name="train",
                                                mode=mode)

        dataset = DCASE_Dataset(files, param)
        
        # data loader and split train and validation
        train_size = int(len(dataset) * (1 - param["fit"]["validation_split"]))
                
        train_loader = DataLoader(Subset(dataset, list(range(0, train_size))),
                                  batch_size=param["fit"]["batch_size"],
                                  shuffle=param["fit"]["shuffle"],
                                  drop_last=True,
                                  num_workers=os.cpu_count(),
                                  pin_memory=True)
        
        val_loader = DataLoader(Subset(dataset, list(range(train_size, len(dataset)))),
                                batch_size=param["fit"]["batch_size"],
                                shuffle=False,
                                drop_last=False,
                                num_workers=os.cpu_count(),
                                pin_memory=True)

        # number of vectors for each wave file
        n_vectors_ea_file = int(len(dataset) / len(files))

        # train model
        epochs = param["fit"]["epochs"]
        loss_func = nn.MSELoss()
        input_channel = param["feature"]["n_mels"] * param["feature"]["n_frames"]
        model = AutoEncoder(input_channel).to(device)
        optimizer = optim.Adam(model.parameters(), lr=param["fit"]["lr"])
        losses = {"train":[], "val":[]}
        print("============== MODEL TRAINING ==============")
        for epoch in range(epochs):
            
            # training
            model.train()
            train_loss = 0
            for batch_idx, data in enumerate(train_loader):
                data = data.to(device, non_blocking=True)
                optimizer.zero_grad()
                reconst = model(data)
                loss = loss_func(data, reconst)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            
            # validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_idx, data in enumerate(val_loader):
                    data = data.to(device)
                    reconst = model(data)
                    loss = loss_func(data, reconst)
                    val_loss += loss.item()
            
            # Avarage loss over mini batches
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            print("Epoch : {} Avarage train_loss : {:.6f}, Avarage val_loss : {:.6f}".format(epoch, train_loss, val_loss))
            losses["train"].append(train_loss)
            losses["val"].append(val_loss)
        
        # calculate y_pred for fitting anomaly score distribution
        y_pred = []
        start_idx = 0
        for file_idx in range(len(files)):
            data = dataset[start_idx : start_idx + n_vectors_ea_file]
            data = data.to(device)
            reconst = model(data)
            mseloss = loss_func(data, reconst)
            y_pred.append(mseloss.item())
            start_idx += n_vectors_ea_file

        # fit anomaly score distribution
        shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(y_pred)
        gamma_params = [shape_hat, loc_hat, scale_hat]
        joblib.dump(gamma_params, score_distr_file_path)
        
        visualizer.loss_plot(losses["train"], losses["val"])
        visualizer.save_figure(history_img)        
        torch.save(model.state_dict(), model_file_path)
        com.logger.info("save_model -> {}".format(model_file_path))
        print("============== END TRAINING ==============")

        del data
        del model