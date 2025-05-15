import numpy as np
import logging
import copy
from itertools import product, combinations as itertools_combinations
from datasets import get_dataset_class

logging.basicConfig(level='WARNING')

OUTPUT_DIR = ''
RESULTS_DIR = ''

def combinations(grid):
    """Returns list of dictionaries for all possible combinations in grid."""
    return list(dict(zip(grid.keys(), values)) for values in product(*grid.values()))

def get_hparams(experiment):
    """Get hyperparameters for a specific experiment."""
    if experiment not in globals():
        raise NotImplementedError(experiment)
    return globals()[experiment].hparams()

def get_script_name(experiment):
    """Get the script name for a specific experiment."""
    if experiment not in globals():
        raise NotImplementedError(experiment)
    return globals()[experiment].fname

#### write experiments here
'''
Experimental order:
- TerraIncognita_ERM_Transfer
- PACS_ERM_Transfer
- VLCS_ERM_Transfer
- WILDSCamelyon_ERM_Transfer
- WILDSFMoW_ERM_Transfer
- CXR_No_Finding_ERM_Transfer
'''

# Constants
N_TRIALS = 100
class CXR_No_Finding_ERM_Transfer:
    fname = 'test.py'

    @staticmethod
    def hparams():
        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [OUTPUT_DIR],
                'python_path': [''],
                'dataset': ['CXR_No_Finding'],
                'results_dir': [RESULTS_DIR],
            },
            'training': {
              'num_epochs': [50000],
              'num_trials': [N_TRIALS],
              'loss_type': ['r'],
              'num_OOD_samples': [10, 20, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 15000, 20000, 25000, 50000, 100000, 250000],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = []
        base_combinations = combinations(flat_grid)

        for combo in base_combinations:
            for train_idxs in [[1,2,3,4]]:
                for test_idx in [0]:
                    if test_idx not in train_idxs:
                            combo_i = combo.copy()
                            combo_i['train_idxs'] = train_idxs
                            combo_i['test_idx'] = test_idx
                            output_combinations.append(combo_i)
        return output_combinations


class TerraIncognita_ERM_Transfer:
    fname = 'test.py'

    @staticmethod
    def hparams():
        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [OUTPUT_DIR],
                'python_path': [''],
                'dataset': ['TerraIncognita'],
                'results_dir': [RESULTS_DIR],
            },
            'training': {
              'num_epochs': [50000],
              'num_trials': [N_TRIALS],
              'loss_type': ['r'],
              'num_OOD_samples': [10, 20, 50, 100, 250, 500, 750, 1000, 2500],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = []
        base_combinations = combinations(flat_grid)

        for combo in base_combinations:
            for train_idxs in [[0,1,2]]:
                for test_idx in [3]:
                    if test_idx not in train_idxs:
                            combo_i = combo.copy()
                            combo_i['train_idxs'] = train_idxs
                            combo_i['test_idx'] = test_idx
                            output_combinations.append(combo_i)

        return output_combinations


class PACS_ERM_Transfer:
    fname = 'test.py'

    @staticmethod
    def hparams():
        num_domains = len(get_dataset_class('PACS').ENVIRONMENTS)
        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [OUTPUT_DIR],
                'python_path': [''],
                'dataset': ['PACS'],
                'results_dir': [RESULTS_DIR],
            },
            'training': {
              'num_epochs': [50000],
              'num_trials': [N_TRIALS],
              'loss_type': ['r'],
              'num_OOD_samples': [10, 20, 50, 100, 250, 500, 750, 1000, 2500],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = []
        base_combinations = combinations(flat_grid)

        for combo in base_combinations:
            for train_idxs in [[0,1,2]]:
                for test_idx in [3]:
                    if test_idx not in train_idxs:
                            combo_i = combo.copy()
                            combo_i['train_idxs'] = train_idxs
                            combo_i['test_idx'] = test_idx
                            output_combinations.append(combo_i)

        return output_combinations

class VLCS_ERM_Transfer:
    fname = 'test.py'

    @staticmethod
    def hparams():
        num_domains = len(get_dataset_class('VLCS').ENVIRONMENTS)
        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [OUTPUT_DIR],
                'python_path': [''],
                'dataset': ['VLCS'],
                'results_dir': [RESULTS_DIR],
            },
            'training': {
              'num_epochs': [50000],
              'num_trials': [N_TRIALS],
              'loss_type': ['r'],
              'num_OOD_samples': [10, 20, 50, 100, 250, 500, 750, 1000, 2500],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = []
        base_combinations = combinations(flat_grid)

        base_cnt = 0
        for combo in base_combinations:
            for train_idxs in [[0,2,3]]:
                for train_idxs in [[0,2,3]]:
                    # for test_idx in range(num_domains):
                    for test_idx in [1]:
                        if test_idx not in train_idxs:
                            combo_i = combo.copy()
                            combo_i['train_idxs'] = train_idxs
                            combo_i['test_idx'] = test_idx
                            output_combinations.append(combo_i)

        return output_combinations

class WILDSCamelyon_ERM_Transfer:
    fname = 'test.py'

    @staticmethod
    def hparams():
        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [OUTPUT_DIR],
                'python_path': [''],
                'dataset': ['WILDSCamelyon'],
                'results_dir': [RESULTS_DIR],
            },
            'training': {
              'num_epochs': [50000],
              'num_trials': [N_TRIALS],
              'loss_type': ['r'],
              'num_OOD_samples': [10, 20, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 15000, 20000, 25000, 50000, 75000, 100000],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = []
        base_combinations = combinations(flat_grid)

        for combo in base_combinations:
            for train_idxs in [[0,1,2]]:
                for test_idx in [3, 4]:
                    if test_idx not in train_idxs:
                            combo_i = combo.copy()
                            combo_i['train_idxs'] = train_idxs
                            combo_i['test_idx'] = test_idx
                            output_combinations.append(combo_i)

        return output_combinations

class WILDSFMoW_ERM_Transfer:
    fname = 'test.py'

    @staticmethod
    def hparams():
        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [OUTPUT_DIR],
                'python_path': [''],
                'dataset': ['WILDSFMoW'],
                'results_dir': [RESULTS_DIR],
            },
            'training': {
              'num_epochs': [50000],
              'num_trials': [N_TRIALS],
              'loss_type': ['r'],
              'num_OOD_samples': [10, 20, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 15000, 20000, 25000, 50000, 75000, 100000, 150000],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = []
        base_combinations = combinations(flat_grid)

        for combo in base_combinations:
            for train_idxs in [[0,1,2]]:
                for test_idx in [3]:
                    if test_idx not in train_idxs:
                        combo_i = combo.copy()
                        combo_i['train_idxs'] = train_idxs
                        combo_i['test_idx'] = test_idx
                        output_combinations.append(combo_i)

        return output_combinations
