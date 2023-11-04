import torch
import copy
import random

from config import get_args
from utils import *

args = get_args()
set_seed(args.seed)

train_dls, test_dl, client_num_samples, traindata_cls_counts, data_distributions = get_dataloader(args)

global_model = get_model(args)
global_model.load_state_dict(torch.load(args.load_path)) if args.load_path is not None else None    # Load saved model if exists
local_model_list = [copy.deepcopy(global_model) for _ in range(args.n_client)]
local_extra_list = [set_model_zero(copy.deepcopy(global_model)) for _ in range(args.n_client)] if args.alg in ('scaffold') else []
global_extra = set_model_zero(copy.deepcopy(global_model)) if args.alg in ('scaffold') else []

best_acc = 0.0

# ========== Federated Learning ==========
print(f"{'='*20} Start Federated Learning {'='*20}")  
for round in range(args.n_round):

    # Choose available clients
    available_clients = sorted(random.sample(range(args.n_client), int(args.sample_fraction * args.n_client)))

    # Federated Process: sync model / local train / global aggregate / global evaluate
    if args.alg == 'fedavg':
        local_train_base(args, train_dls, local_model_list, global_model, available_clients)
        aggregate_model(global_model, local_model_list, client_num_samples, available_clients)
        acc, best_acc = evaluation(args, global_model, test_dl, best_acc)
    elif args.alg == 'fedprox':
        local_train_fedprox(args, train_dls, local_model_list, global_model, available_clients)
        aggregate_model(global_model, local_model_list, client_num_samples, available_clients)
        acc, best_acc = evaluation(args, global_model, test_dl, best_acc)
    elif args.alg == 'scaffold':
        local_train_scaffold(args, train_dls, local_model_list, global_model, available_clients, local_extra_list, global_extra)
        aggregate_model(global_model, local_model_list, client_num_samples, available_clients)
        acc, best_acc = evaluation(args, global_model, test_dl, best_acc)

    print('>> Round {} | Current Acc: {:.5f}, Best Acc: {:.5f} <<\n{}'.format(round, acc, best_acc, '-'*80))