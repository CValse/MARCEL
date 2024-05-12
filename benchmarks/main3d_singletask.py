import os
import torch
import torch.nn.functional as F
import gc
import pip

import argparse
from dataclasses import asdict

from sklearn.metrics import r2_score
import pandas as pd
import torch.nn as nn

from pytorch_lightning import LightningModule, Trainer, LightningDataModule

###################################################
import os
import numpy as np
#os.environ['CUDA_VISIBLE_DEVICES'] = gpus
torch.cuda.empty_cache()
#print(os.environ['CUDA_VISIBLE_DEVICES'])

seed = 42
torch.manual_seed(seed)

# If you are using a GPU, you should also set the seed for CUDA operations
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

np.random.seed(seed)
################################################## 

def r2_score_torch(y_true, y_pred):
    """
    Compute the R-squared score.

    Parameters:
        y_true (torch.Tensor): The true target values.
        y_pred (torch.Tensor): The predicted target values.

    Returns:
        float: The R-squared score.
    """
    # Calculate the mean of the true target values
    y_mean = torch.mean(y_true)
    # Calculate the total sum of squares (TSS)
    tss = torch.sum((y_true - y_mean) ** 2)
    # Calculate the residual sum of squares (RSS)
    rss = torch.sum((y_true - y_pred) ** 2)
    # Calculate R-squared score
    r2 = 1 - rss / tss

    return r2.item()

class DataModule(LightningDataModule):
    def __init__(self, hparams, dataset=None, multitask = False):
        super().__init__()
        self.hparams.update(hparams.__dict__) if hasattr(hparams, "__dict__") else self.hparams.update(hparams)
        self._saved_dataloaders = dict()
        self.dataset = dataset
        self.multitask = multitask
        
        if self.dataset is None:
            self.variable_name = None
            unique_variables = 1

            if self.hparams.dataset == 'Drugs':
                dataset = Drugs('/mnt/data/MARCEL/datasets/Drugs', max_num_conformers=self.hparams.max_num_conformers).shuffle()
            elif self.hparams.dataset == 'Kraken':
                dataset = Kraken('/mnt/data/MARCEL/datasets/Kraken', max_num_conformers=self.hparams.max_num_conformers).shuffle()
            elif self.hparams.dataset == 'BDE':
                dataset = BDE('/mnt/data/MARCEL/datasets/BDE').shuffle()
                self.variable_name = 'is_ligand'
                unique_variables = 2
            elif self.hparams.dataset == 'EE':
                dataset = EE('/mnt/data/MARCEL/datasets/EE', max_num_conformers=self.hparams.max_num_conformers).shuffle()
                self.variable_name = 'config_id'
                unique_variables = 2

            if self.multitask:
                self.hparams.target = 'all'
                pass
            else:
                #autoscaling
                target_id = dataset.descriptors.index(self.hparams.target)
                dataset.y = dataset.y[:, target_id]
                #mean = dataset.y.mean(dim=0, keepdim=True)
                #std = dataset.y.std(dim=0, keepdim=True)
                #dataset.y = ((dataset.y - mean) / std).to('cuda')
                #mean = mean.to('cuda')
                #std = std.to('cuda')
            
                #data.dataset.data.y = data.dataset.y
            
            self.dataset = dataset
            self.max_atomic_num = self.dataset.data.x[:, 0].max().item() + 1
            self.unique_variables = unique_variables
            print('--done---')

    def split_compute(self):

        split = self.dataset.get_idx_split(train_ratio=self.hparams.train_ratio, 
                                      valid_ratio=self.hparams.valid_ratio, 
                                      seed=self.hparams.seed)
        self.train_dataset = self.dataset[split['train']]
        self.valid_dataset = self.dataset[split['valid']]
        self.test_dataset = self.dataset[split['test']]

        print(f'{len(self.train_dataset)} training data, {len(self.test_dataset)} test data and {len(self.valid_dataset)} validation data')

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self._get_dataloader(self.valid_dataset, "val")

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")
    
    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        store_dataloader = store_dataloader
        
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if self.hparams.model3d_augmentation:
            strategy = 'random'
        else:
            strategy = 'first'
            
        if stage == "train":
            shuffle=True                              
        else:
            shuffle=False
            if stage == "train"=='test':
                strategy = 'first'

        if self.variable_name is None:
            dl = DataLoader(dataset, batch_sampler=EnsembleSampler(dataset, 
                                                                   batch_size=self.hparams.batch_size, 
                                                                   strategy=strategy, 
                                                                   shuffle=shuffle),
                           num_workers=20)
        else:
            dl = MultiBatchLoader(dataset, batch_sampler=EnsembleMultiBatchSampler(dataset, 
                                                                                   batch_size=self.hparams.batch_size, 
                                                                                   strategy=strategy, 
                                                                                   shuffle=shuffle, 
                                                                                   variable_name=self.variable_name),
                                 num_workers=20)
        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl

import math
def compute_pnorm(model: nn.Module) -> float:
    """
    Computes the norm of the parameters of a model.
    :param model: A PyTorch model.
    :return: The norm of the parameters of the model.
    """
    return math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters() if p.requires_grad]))


def compute_gnorm(model: nn.Module) -> float:
    """
    Computes the norm of the gradients of a model.
    :param model: A PyTorch model.
    :return: The norm of the gradients of the model.
    """
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))

class ModelLM(LightningModule):
    def __init__(self, max_atomic_num=None, whole_dataset = None, unique_variables=1, **kwargs):
        super().__init__()
        #self.kwargs.update(kwargs.__dict__) if hasattr(kwargs, "__dict__") else self.kwargs.update(kwargs)
        print(kwargs.get('model3d'))
        if kwargs.get('model3d').model == 'SchNet':
            model_factory = lambda: SchNet(max_atomic_num=max_atomic_num, 
                                           **asdict(kwargs.get('model3d').schnet))
        elif kwargs.get('model3d').model == 'DimeNet':
            model_factory = lambda: DimeNet(max_atomic_num=max_atomic_num, 
                                            **asdict(kwargs.get('model3d').dimenet))
        elif kwargs.get('model3d').model == 'DimeNet++':
            model_factory = lambda: DimeNetPlusPlus(max_atomic_num=max_atomic_num, 
                                                    **asdict(kwargs.get('model3d').dimenetplusplus))
        elif kwargs.get('model3d').model == 'GemNet':
            model_factory = lambda: GemNetT(max_atomic_num=max_atomic_num, 
                                            **asdict(kwargs.get('model3d').gemnet))
        elif kwargs.get('model3d').model == 'ChIRo':
            model_factory = lambda: ChIRo(**asdict(kwargs.get('model3d').chiro))
            
        elif kwargs.get('model3d').model == 'PaiNN':
            model_factory = lambda: PaiNN(max_atomic_num=max_atomic_num, 
                                          **asdict(kwargs.get('model3d').painn))
        elif kwargs.get('model3d').model == 'ClofNet':
            model_factory = lambda: ClofNet(max_atomic_num=max_atomic_num, 
                                            **asdict(kwargs.get('model3d').clofnet))
        elif kwargs.get('model3d').model == 'LEFTNet':
            model_factory = lambda: LEFTNet(max_atomic_num=max_atomic_num, 
                                            **asdict(kwargs.get('model3d').leftnet))
        elif kwargs.get('model3d').model == 'ChytorchDiscrete':
            model_factory = lambda: ChytorchDiscrete(max_neighbors=max_atomic_num, 
                                                     **asdict(kwargs.get('model3d').chytorch_discrete))
        elif kwargs.get('model3d').model == 'ChytorchConformer':
            model_factory = lambda: ChytorchConformer(**asdict(kwargs.get('model3d').chytorch_conformer))
            
        elif kwargs.get('model3d').model == 'ChytorchRotary':
            model_factory = lambda: ChytorchRotary(max_neighbors=max_atomic_num, 
                                                   **asdict(kwargs.get('model3d').chytorch_rotary))
        self.device_= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Model3D(model_factory, 
                           hidden_dim=kwargs.get('hidden_dim'), 
                           out_dim=1,
                           unique_variables=unique_variables, 
                           device='cuda').to('cuda')
        
        self.loss_fn = nn.MSELoss() #LOGITS #GroupedScaledMAELoss(torch.ones(4, dtype=torch.long))
        #self.loss_fn = GroupedScaledMAELoss(torch.ones(20, dtype=torch.long))
        
        self.lr = kwargs.get('learning_rate')
        self.wd = kwargs.get('learning_rate')
        self.whole_dataset = whole_dataset

        self._reset_losses_dict()
        self._reset_inference_results()
        self.save_hyperparameters(ignore=["cosine_annealing_lr","linear_warmup_cosine_annealing_lr",
                                          "model1d","model2d","model3d","model4d","modelfprf","one_cycle_lr",
                                          "reduce_lr_on_plateau","whole_dataset","device","scheduler"])

    def forward(self, batch):
        out = self.net(batch)
        return out
        
    def configure_optimizers(self):
        o = AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        s = CyclicLR(o, self.lr, 2e-4, 1000, mode='triangular', cycle_momentum=False)
        # instantiate the WeakMethod in the lr scheduler object into the custom scale function attribute
        #s._scale_fn_custom = s._scale_fn_ref()
        # remove the reference so there are no more WeakMethod references in the object
        #s._scale_fn_ref = None
        return [o], [{'scheduler': s, 'interval': 'step', 'name': 'lr_scheduler'}]

    def training_step(self, batch, batch_idx):
        pnorm = compute_pnorm(self.net)
        gnorm = compute_gnorm(self.net)
        self.log(f'(training) pnorm', pnorm)
        self.log(f'(training) gnorm', gnorm)
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")
        
    def step(self, batch, stage):
        start = time()
        if type(batch) is not list:
            batch = [batch]
        molecule_idx = batch[0].molecule_idx.to('cuda')
        dataset = self.whole_dataset.y.to('cuda')
        targets = dataset[molecule_idx].squeeze()

        with torch.set_grad_enabled(stage == "train"):
            pred = self(batch)
            loss = self.loss_fn(pred.squeeze(), targets)
            #loss = self.loss_fn(pred.squeeze(), targets, prompts-121)
            
            if stage == "test":
                self.inference_results['y_pred'].append(pred.squeeze())
                self.inference_results['y_true'].append(targets.squeeze())
                return None

            r2=r2_score_torch(targets.cpu(),pred.squeeze().cpu().detach())

            self.logging_info[f'{stage}_loss'].append(loss.item())
            self.logging_info[f'{stage}_r2'].append(r2)
            self.logging_info[f'{stage}_time'].append(time()-start)
            
            if stage == 'train':
                self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
                self.log(f'{stage}_step_loss', loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
            #model.save_checkpoint('checkpoint.pth')
            return loss
        
    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:                    
            result_dict = {
                "epoch": float(self.current_epoch),
                "train_epoch_loss": torch.tensor(self.logging_info["train_loss"]).mean().item(),
                "train_epoch_r2": torch.tensor(self.logging_info["train_r2"]).mean().item(),
                "val_epoch_loss": torch.tensor(self.logging_info["val_loss"]).mean().item(),
                "val_epoch_r2": torch.tensor(self.logging_info["val_r2"]).mean().item(),
                "train_epoch_time": sum(self.logging_info["train_time"]),
                "val_epoch_time": sum(self.logging_info["val_time"]),
                }
            self.log_dict(result_dict, logger=True, sync_dist=True)
            
        self._reset_losses_dict()
    
    def on_test_epoch_end(self) -> None:
        for key in self.inference_results.keys():
            self.inference_results[key] = torch.cat(self.inference_results[key], dim=0)
    
    def _reset_losses_dict(self):
        self.logging_info = {
            "train_loss": [],
            "train_r2": [], 
            "train_mse": [], 
            "val_loss": [],
            "val_r2": [],
            "train_sample_size": [], 
            "val_sample_size": [],
            "train_time": [],
            "val_time": [],
        }
        
    def _reset_inference_results(self):
        self.inference_results = {'y_pred': [],
                                  'y_true': []}

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

def install_torch(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package, '-f', 'https://data.pyg.org/whl/torch-2.3.0+cu121.html'])
    else:
        pip._internal.main(['install', package, '-f', 'https://data.pyg.org/whl/torch-2.3.0+cu121.html'])

def parse_args():
    parser = argparse.ArgumentParser(description="Your script description here")
    parser.add_argument("--dataname", type=str, default="BDE", help="Whether to consider chirality")
    parser.add_argument("--target", type=str, default="BindingEnergy", help="Type of chirality")
    parser.add_argument("--modeltype", type=str, default="ClofNet", help="Masked fraction")
    return parser.parse_args()

def config_to_dict(config_class):
    config_dict = {}
    for attr_name in dir(config_class):
        if not attr_name.startswith("__") and not callable(getattr(config_class, attr_name)):
            config_dict[attr_name] = getattr(config_class, attr_name)
    return config_dict

def find_best_model_checkpoint(directory_path):
    for filename in os.listdir(directory_path):
        if 'best-model' in filename:
            return filename
    return "No checkpoint file containing 'best_model' found."  

def main():
    install_torch('torch-geometric')
    args = parse_args()
    dataname = args.dataname
    target = args.target
    modeltype = args.modeltype

    config = Config
    config.dataset = dataname
    config.target = target
    config.device = 'cuda:0'
    
    
    ######3DMODEL
    config.model3d.model=modeltype
    config.model3d.augmentation = True

    config_dict = config_to_dict(config)
    subkeys = ["dataset",
               "max_num_conformers",
               "target",
               "train_ratio",
               "valid_ratio",
               "seed",
               "model3d", #.augmentation"
               "batch_size"]
    config_dict_datamodule = {}
    for k,v in config_dict.items():
        if k in subkeys:
            if k=="model3d":
                config_dict_datamodule[f'{k}_augmentation']=config_dict[k].augmentation
            else:
                config_dict_datamodule[k]=v

    data = DataModule(config_dict_datamodule)
    data.prepare_data()
    data.split_compute()

    model = ModelLM(max_atomic_num=data.max_atomic_num, 
                    whole_dataset = data.dataset, 
                    unique_variables=data.unique_variables, **config_dict)
    print(f'#PARAMS = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    dir_name = f"log_{dataname}_{target}_{modeltype}_v0"

    dir_load_model = None
    log_dir_folder = '/mnt/code/logs/'
    log_dir_folder = os.path.join(log_dir_folder, dir_name)
    if os.path.exists(log_dir_folder):
        if os.path.exists(os.path.join(log_dir_folder, "last.ckpt")):
            dir_load_model = os.path.join(log_dir_folder, "last.ckpt")
        csv_path = os.path.join(log_dir_folder, "metrics.csv")
        while os.path.exists(csv_path):
            csv_path = csv_path + '.bak'
        if os.path.exists(os.path.join(log_dir_folder, "metrics.csv")):
            os.rename(os.path.join(log_dir_folder, "metrics.csv"), csv_path)

    metric_to_monitor = "val_epoch_loss"

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir_folder,
        monitor=metric_to_monitor,
        mode = 'min',
        save_top_k=1,
        save_last=True,
        every_n_epochs=5,
        save_weights_only=True,
        verbose=True,
        filename="best-model-{epoch}-{val_epoch_loss:.4f}",
    )
    
    
    early_stopping = early_stop_callback = EarlyStopping(
            monitor=metric_to_monitor,  # The metric you want to monitor
            patience=config.patience,  # Number of epochs with no improvement after which training will be stopped
            verbose=True,
            mode='min'  # Minimizing the validation loss
        )
    
    tb_logger = TensorBoardLogger(log_dir_folder, name="tensorbord")#, version="", default_hp_metric=False)
    csv_logger = CSVLogger(log_dir_folder, name="", version="")
    
    model_params = dict(
        devices=1, #args['ngpus'],
        accelerator='gpu', #args['accelerator'],
        default_root_dir=log_dir_folder, #args['log_dir'],
        logger=[tb_logger, csv_logger],
        enable_progress_bar=True)
    
    
    model_params.update(dict(
        max_epochs=config.num_epochs,#1000,
        callbacks=[checkpoint_callback, early_stopping],
        #enable_checkpointing=False,
        gradient_clip_val=10,#args['clip_norm'],
        #precision="16-mixed",
    ))

    torch.set_float32_matmul_precision('medium')

    trainer = pl.Trainer(**model_params)
    trainer.fit(model, datamodule=data, ckpt_path=dir_load_model)

    # Call the function and print the result
    best_model_checkpoint_filename = find_best_model_checkpoint(log_dir_folder)
    checkpoint_path = os.path.join(log_dir_folder, best_model_checkpoint_filename)  #f'{log_dir_folder}/'+best_model_checkpoint_filename

    #EVALUATE TEST
    test_trainer = pl.Trainer(
        max_epochs=-1,
        num_nodes=1,
        default_root_dir=checkpoint_path,
        logger=False,
        accelerator='gpu',
        devices=1,
        enable_progress_bar=True,
    )

    test_trainer.test(model=model, ckpt_path=checkpoint_path, datamodule=data)
    perf_dict = {'y_true': model.inference_results['y_true'].cpu().numpy(), 
                 'y_pred': model.inference_results['y_pred'].cpu().numpy()}
    r2test = r2_score(perf_dict['y_true'], perf_dict['y_pred'])
    #mae_test = mean_absolute_error(perf_dict['y_true'], perf_dict['y_pred'])
    pd.DataFrame(perf_dict).to_csv(log_dir_folder+f'/test_pred_ba_chi_{str(r2test)[:4]}.csv')
    print(f"R2 = {str(r2test)[:4]}")
    perf_dict=[]
    torch.cuda.empty_cache()
    gc.collect()
    
    
if __name__ == '__main__':
    install_torch('torch-scatter')
    install_torch('torch-sparse')
    install_torch('torch-cluster')
    install_torch('torch-geometric')
    install('ase')

    from torch.nn.utils import clip_grad_norm_
# from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR

    from torch.utils.tensorboard import SummaryWriter
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
    
    from config import Config
    from data.ee import EE
    from data.bde import BDE
    from data.drugs import Drugs
    from data.kraken import Kraken
    from happy_config import ConfigLoader
    from loaders.samplers import EnsembleSampler, EnsembleMultiBatchSampler
    from loaders.multibatch import MultiBatchLoader
    from utils.early_stopping import EarlyStopping, generate_checkpoint_filename
    
    from models.model_3d import Model3D
    from models.models_3d.chiro import ChIRo
    from models.models_3d.painn import PaiNN
    from models.models_3d.schnet import SchNet
    from models.models_3d.gemnet import GemNetT
    from models.models_3d.dimenet import DimeNet, DimeNetPlusPlus
    from models.models_3d.clofnet import ClofNet
    from models.models_3d.leftnet import LEFTNet
    from models.models_3d.chytorch_discrete import ChytorchDiscrete
    #from models.models_3d.chytorch_conformer import ChytorchConformer
    
    import pickle
    from time import time
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    import pytorch_lightning as pl
    
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
    from torch_geometric.loader import DataLoader


    main()

    
    

    