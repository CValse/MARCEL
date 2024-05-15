import pip
import gc
import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
import torch.nn as nn

import torch
import torch.nn.functional as F

from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
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

class Molecules(Dataset):
    def __init__(self, smiles_ids, attention_masks, labels, fingerprint=None, input_type='smiles'):
        self.smiles_ids = smiles_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.fingerprint = fingerprint
        self.input_type = input_type

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        if self.input_type == 'SMILES':
            smiles = self.smiles_ids[index]
            attention_mask = self.attention_masks[index]
            y = self.labels[index]
            return smiles, attention_mask, y.clone()
        else:
            fingerprint = self.fingerprint[index]
            y = self.labels[index]
            return torch.tensor(fingerprint, dtype=torch.long), y.clone()

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
    def __init__(self, hparams, dataset=None):
        super().__init__()
        self.hparams.update(hparams.__dict__) if hasattr(hparams, "__dict__") else self.hparams.update(hparams)
        self._saved_dataloaders = dict()
        self.dataset = dataset
        
        if self.dataset is None:
            variable_name = None
            unique_variables = 1
            
            if self.hparams.dataset == 'Drugs':
                dataset = Drugs('/mnt/data/MARCEL/datasets/Drugs', max_num_conformers=self.hparams.max_num_conformers).shuffle()
            elif self.hparams.dataset == 'Kraken':
                dataset = Kraken('/mnt/data/MARCEL/datasets/Kraken', max_num_conformers=self.hparams.max_num_conformers).shuffle()
            elif self.hparams.dataset == 'BDE':
                dataset = BDE('/mnt/data/MARCEL/datasets/BDE').shuffle()
                variable_name = 'is_ligand'
                unique_variables = 2
            elif self.hparams.dataset == 'EE':
                dataset = EE('/mnt/data/MARCEL/datasets/EE', max_num_conformers=self.hparams.max_num_conformers).shuffle()
                variable_name = 'config_id'
                unique_variables = 2
            
            #autoscaling
            target_id = dataset.descriptors.index(self.hparams.target)
            labels = dataset.y[:, target_id]
            mean = labels.mean(dim=0).item()
            std = labels.std(dim=0).item()
            labels = (labels - mean) / std
            
            if variable_name is not None:
                smiles = concatenate_smiles(dataset, variable_name)
            else:
                smiles = construct_smiles(dataset)
            fingerprint = construct_fingerprint(smiles) if self.hparams.model1d_input_type == 'Fingerprint' else None

            tokenizer = RobertaTokenizer.from_pretrained('seyonec/PubChem10M_SMILES_BPE_450k')
            dicts = tokenizer(smiles, return_tensors='pt', padding='longest')
            smiles_ids, attention_masks = dicts['input_ids'], dicts['attention_mask']
            vocab_size = tokenizer.vocab_size if self.hparams.model1d_input_type == 'SMILES' else fingerprint.shape[1]

            dataset = Molecules(smiles_ids, attention_masks, labels, fingerprint, input_type=self.hparams.model1d_input_type)

            self.dataset = dataset
            self.vocab_size=vocab_size
            self.tokenizer=tokenizer
            self.smiles_ids=smiles_ids
            #modelnet = model.to(device)

            print('--done---')

    def split_compute(self):
        train_ratio = self.hparams.train_ratio
        valid_ratio = self.hparams.valid_ratio
        test_ratio = 1 - train_ratio - valid_ratio

        train_len = int(train_ratio * len(self.dataset))
        valid_len = int(valid_ratio * len(self.dataset))
        test_len = len(self.dataset) - train_len - valid_len

        self.train_dataset, self.valid_dataset, self.test_dataset = random_split(self.dataset, lengths=[train_len, valid_len, test_len])
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

        if stage == "train":
            dl = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers = 20)                                  
        else:
            dl = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers = 20) 

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

class Model1D(LightningModule):
    def __init__(self, vocab_size=None, tokenizer=None, smiles_ids=None, **hparams):
        super().__init__()
        #self.hparams.update(hparams.__dict__) if hasattr(hparams, "__dict__") else self.hparams.update(hparams)
        
        if hparams.get('model1d').model == 'LSTM':
            self.net = LSTM(
                vocab_size, hparams.get('hidden_dim'), hparams.get('hidden_dim'), 1,
                hparams.get('model1d').num_layers, hparams.get('dropout'), padding_idx=tokenizer.pad_token_id)
        elif hparams.get('model1d').model == 'Transformer':
            self.net = Transformer(
                vocab_size, hparams.get('model1d').embedding_dim, smiles_ids.shape[1],
                hparams.get('model1d').num_heads, hparams.get('hidden_dim'), 1,
                hparams.get('model1d').num_layers, hparams.get('dropout'), padding_idx=tokenizer.pad_token_id)
                
        self.loss_fn = nn.MSELoss() #LOGITS #GroupedScaledMAELoss(torch.ones(4, dtype=torch.long))
        self.device_= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_type = hparams.get('model1d').input_type
        self.lr = hparams.get('learning_rate')
        self.wd = hparams.get('weight_decay')

        self._reset_losses_dict()
        self._reset_inference_results()
        self.save_hyperparameters(ignore=["cosine_annealing_lr","linear_warmup_cosine_annealing_lr",
                                          "model1d","model2d","model3d","model4d","modelfprf","one_cycle_lr",
                                          "reduce_lr_on_plateau","whole_dataset","device","scheduler"])

    def forward(self, batch):
        if self.input_type == 'SMILES':
            input_ids, attention_mask, y = batch
            if isinstance(self.net, Transformer):
                out = self.net(input_ids, attention_mask)
            else:
                out = self.net(input_ids)
        else:
            fingerprints, y = batch
            out = self.net(fingerprints)
        return out, y
        
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

        with torch.set_grad_enabled(stage == "train"):
            pred, targets = self(batch)
            loss = self.loss_fn(pred.squeeze(), targets)
            
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
    parser.add_argument("--input_type", type=str, default="SMILES", help="Masked fraction")
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
    input_type = args.input_type
    modeltype = args.modeltype

    config = Config
    config.dataset = dataname
    config.target = target
    config.device = 'cuda:0'
    
    #config.max_num_conformers = 20
    config.model1d.input_type = input_type
    config.model1d.model = modeltype


    config_dict = config_to_dict(config)
    subkeys = ["dataset",
               "max_num_conformers",
               "target",
               "train_ratio",
               "valid_ratio",
               "seed",
               "model1d", #.augmentation"
               "batch_size"]
    config_dict_datamodule = {}
    for k,v in config_dict.items():
        if k in subkeys:
            if k=="model1d":
                config_dict_datamodule[f'{k}_input_type']=config_dict[k].input_type
            else:
                config_dict_datamodule[k]=v

    data = DataModule(config_dict_datamodule)
    data.prepare_data()
    data.split_compute()

    model = Model1D(vocab_size=data.vocab_size, tokenizer=data.tokenizer, smiles_ids=data.smiles_ids, **config_dict)

    print(f'#PARAMS = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    dir_name = f"log_1D_{dataname}_{target}_{modeltype}_{input_type}_v0"

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
    pd.DataFrame(perf_dict).to_csv(log_dir_folder+f'/test_pred_r2_{str(r2test)[:4]}.csv')
    print(f"R2 = {str(r2test)[:4]}")
    perf_dict=[]
    torch.cuda.empty_cache()
    gc.collect()
    
    
if __name__ == '__main__':
    install_torch('torch-scatter')
    install_torch('torch-sparse')
    install_torch('torch-cluster')
    install_torch('torch-geometric')
    install('transformers')

    
    from transformers import RobertaTokenizer
    #from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
    
    from config import Config
    from data.ee import EE
    from data.bde import BDE
    from data.drugs import Drugs
    from data.kraken import Kraken
    from happy_config import ConfigLoader
    from utils.early_stopping import EarlyStopping, generate_checkpoint_filename
    
    from models.model_1d import LSTM, Transformer
    from models.models_1d.utils import construct_fingerprint, construct_smiles, concatenate_smiles
    #from train_1d import *
    from time import time
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    import pytorch_lightning as pl


    main()

    
    

    