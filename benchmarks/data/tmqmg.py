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
    descriptors = ['gibbs_energy', 'tzvp_lumo_energy', 'tzvp_homo_energy', 'tzvp_homo_lumo_gap','homo_lumo_gap_delta','tzvp_electronic_energy','electronic_energy_delta','tzvp_dispersion_energy','dispersion_energy_delta','enthalpy_energy','gibbs_energy','heat_capacity','entropy','tzvp_dipole_moment','dipole_moment_delta','polarisability','lowest_vibrational_frequency','highest_vibrational_frequency']

    def __init__(self, root, max_num_conformers=None, transform=None, pre_transform=None, pre_filter=None, chiro_data = False):
        self.max_num_conformers = max_num_conformers
        self.chiro_data=chiro_data
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
        with Chem.SDMolSupplier(raw_file, removeHs=False) as suppl:
            for idx, mol in enumerate(tqdm(suppl)):
                #id_ = mol.GetProp('ID')
                name = mol.GetProp('_Name')
                smiles = mol.GetProp('smiles')

                if self.chiro_data:
                    data = mol_to_data_chiro(mol)
                else:
                    data = mol_to_data_obj(mol)
                data.name = name
                data.id = idx

                data.smiles = smiles
                data.y = []
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

        label_file = raw_file.replace('.sdf', '.txt')
        labels = pd.read_csv(label_file)
        
        print(labels)

        for name, mol_list in tqdm(mols.items()):
            row = labels[labels['name'] == name]
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