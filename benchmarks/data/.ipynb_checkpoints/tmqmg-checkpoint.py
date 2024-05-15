import torch
import pandas as pd

from tqdm import tqdm
from rdkit import Chem
from collections import defaultdict
from torch_geometric.data import extract_zip

from loaders.utils import mol_to_data_obj
from loaders.utils_chiro import mol_to_data_chiro
from loaders.ensemble import EnsembleDataset


class tmQMg(EnsembleDataset):

    def __init__(self, root, max_num_conformers=None, transform=None, pre_transform=None, pre_filter=None, chiro_data = False):
        self.max_num_conformers = max_num_conformers
        self.chiro_data=chiro_data
        self.descriptors =['tzvp_lumo_energy','tzvp_homo_energy','tzvp_homo_lumo_gap','homo_lumo_gap_delta','tzvp_electronic_energy','electronic_energy_delta','tzvp_dispersion_energy','dispersion_energy_delta','enthalpy_energy','enthalpy_energy_correction','gibbs_energy','gibbs_energy_correction','zpe_correction','heat_capacity','entropy','tzvp_dipole_moment','dipole_moment_delta','polarisability','lowest_vibrational_frequency','highest_vibrational_frequency']

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return 'tmQMgEnsemble_processed.pt' if self.max_num_conformers is None \
            else f'tmQMgEnsemble_processed_{self.max_num_conformers}.pt'

    @property
    def raw_file_names(self):
        return 'tmQMg.zip'

    @property
    def num_molecules(self):
        return self.y.shape[0]

    @property
    def num_conformers(self):
        return len(self)

    def process(self):
        data_list = []
        quantities = self.descriptors

        mols = defaultdict(list)
        raw_file = self.raw_paths[0]
        extract_zip(raw_file, self.raw_dir)
        raw_file = raw_file.replace('.zip', '.sdf')

        label_file = raw_file.replace('.sdf', '.txt')
        labels = pd.read_csv(label_file)
        
        with Chem.SDMolSupplier(raw_file, removeHs=False) as suppl:
            for idx, mol in enumerate(tqdm(suppl)):
                #id_ = mol.GetProp('ID')
                name = mol.GetProp('_Name')
                name = name.replace('.xyz','')
                if name in labels['name'].to_list():
                    smiles = mol.GetProp('smiles')
    
                    if self.chiro_data:
                        data = mol_to_data_chiro(mol)
                    else:
                        data = mol_to_data_obj(mol)
                    data.name = name
                    data.id = idx
    
                    data.smiles = smiles

                    row = labels[labels['name'] == name]
                    data.y = torch.Tensor([row[quantity].item() for quantity in quantities]).unsqueeze(0)
                    #for quantity in quantities:
                    #    data.y.append(float(mol.GetProp(quantity)))
                    #    # setattr(data, quantity, float(mol.GetProp(quantity)))
                    #data.y = torch.Tensor(data.y).unsqueeze(0)
    
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
    
                    mols[name].append(data)

        # molecule_idx = []
        cursor = 0
        y = []
        
        #print(labels.columns)
        #print('---------')
        #print(quantities)

        for name, mol_list in tqdm(mols.items()):
            name = name.replace('.xyz','')
            if name in labels['name'].to_list():
                row = labels[labels['name'] == name]
                #print(name)
                #print(row)
                y.append(torch.Tensor([row[quantity].item() for quantity in quantities]))
    
                if self.max_num_conformers is not None:
                    # sort energy and take the lowest energy conformers
                    mol_list = sorted(mol_list, key=lambda x: x.y[:, quantities.index('gibbs_energy')].item())
                    mol_list = mol_list[:self.max_num_conformers]
    
                # molecule_idx += [cursor] * len(mol_list)
                for mol in mol_list:
                    mol.molecule_idx = cursor
                    data_list.append(mol)
                cursor += 1

        y = torch.stack(y, dim=0)
        # molecule_idx = torch.Tensor(molecule_idx).long()
        data, slices = self.collate(data_list)
        torch.save((data, slices, y), self.processed_paths[0])