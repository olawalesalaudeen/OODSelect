import argparse
import sys
sys.path.append('../src')
import os
from scipy.stats import pearsonr
from gurobi_solver import GurobiSolver
from test_utils import load_prediction_data, prepare_dataset, extract_data, probit_transform
import torch

parser = argparse.ArgumentParser(description='Run Gurobi solver with grid search')
parser.add_argument('--N', type=int, required = True)
parser.add_argument('--dataset', type=str, default='TerraIncognita')
parser.add_argument('--num_domains', type=int, default=4)
parser.add_argument('--train_idxs', type=str, default='0-1-2')
parser.add_argument('--test_idx', type=str, default='3')
parser.add_argument('--results_dir', type=str,
                    default="")
parser.add_argument('--time_limit', type=int, default=300)
parser.add_argument('--output_dir', type=str, default='')
args = parser.parse_args()

print('Args:')
for k, v in sorted(vars(args).items()):
    print('\t{}: {}'.format(k, v))

X, y = extract_data(*load_prediction_data(args.dataset, args.num_domains,
                                        args.results_dir, args.train_idxs, args.test_idx))
train_loader, val_loader, test_loader, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = prepare_dataset(X, y)

solver = GurobiSolver(X_train_tensor, probit_transform(y_train_tensor), time_limit=args.time_limit, disp=1)
selected_idxs = solver.solve(args.N)

selection = torch.zeros((X.shape[1], ))
selection[selected_idxs] = 1

torch.save(selection, os.path.join(args.output_dir, f'Test_{args.dataset}_{args.train_idxs}_{args.test_idx}_{args.N}.pt'))

train_pearson = pearsonr(probit_transform(y_train_tensor), probit_transform(X_train_tensor[:, selected_idxs].sum(axis = 1)/len(selected_idxs)))[0]
test_pearson = pearsonr(probit_transform(y_test_tensor), probit_transform(X_test_tensor[:, selected_idxs].sum(axis = 1)/len(selected_idxs)))[0]

print(f'Train Pearson: {train_pearson}, Test Pearson: {test_pearson}')
