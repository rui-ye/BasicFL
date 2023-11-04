import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--save_model', action="store_true", help="whether to save the global model.")
    parser.add_argument('--local_eval', action="store_true", help="whether to evaluate local model.")
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--datadir', type=str, default="./data/", help="Data directory")

    # Training
    parser.add_argument('--model', type=str, default='resnet20_cifar', help='neural network used in training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--n_iteration', type=int, default=100, help='number of local iterations')

    # FL Setting
    parser.add_argument('--alg', type=str, default='fedavg', help='federated learning algorithm to run')
    parser.add_argument('--n_client', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--n_round', type=int, default=200, help='number of maximum communication roun')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--skew_class', type=int, default=2, help='The parameter for the noniid-skew for data partitioning')

    # For FEMNIST dataset
    parser.add_argument('--femnist_sample_top', type=int, default=1, help='whether to sample top clients from femnist')
    parser.add_argument('--femnist_train_num', type=int, default=20, help='how many clients from femnist are sampled')
    parser.add_argument('--femnist_test_num', type=int, default=20, help='number of testing clients from femnist')

    # For FLAIR dataset
    parser.add_argument('--flair_num_test_client', default=10, type=int, help='num of client for testing in the flair dataset')
    parser.add_argument('--flair_img_size',type=int, default=64,help='flair image cropped size')
    parser.add_argument('--flair_client_select', type=str, default='rand',help='select client according to its num of sample')
    parser.add_argument('--flair_gen_class', type=int, default=153, help='generated classes')
    parser.add_argument('--flair_num_per_class', type=int, default=1, help='generated image per classes')
    parser.add_argument('--flair_syn_lr', type=float, default=0.01, help='distillation loss weight')

    # FedAvgM
    parser.add_argument('--server_momentum', type=float, default=0.0)

    # FedProx / MOON
    parser.add_argument('--mu', type=float, default=0.01)

    # FedDecorr
    parser.add_argument('--decorr_beta', type=float, default=0.1, help='parameter for loss term in Feddecor')

    # FedExP
    parser.add_argument('--exp_eps', type=float, default=1e-3, help='parameter for FedEXP in model aggregation')

    # FedDyn
    parser.add_argument('--dyn_alpha', type=float, default=0.01, help='parameter for FedDyn')

    # FedADAM
    parser.add_argument('--adam_server_momentum_1', type=float, default=0.9, help='first order parameter for fedadam.')
    parser.add_argument('--adam_server_momentum_2', type=float, default=0.99, help='second order parameter for fedadam.')
    parser.add_argument('--adam_server_lr', type=float, default=1.0, help='server learning rate for fedadam.')
    parser.add_argument('--adam_tau', type=float, default=0.001, help='tau for fedadam.')

    # FedSAM
    parser.add_argument('--sam_rho', type=float, default=0.05, help='rho for fedsam.')

    # VHL
    parser.add_argument('--VHL_alpha', default=1.0, type=float)
    parser.add_argument('--VHL_feat_align', action="store_true", help='if aligning feature in training')
    parser.add_argument('--VHL_generative_dataset_root_path', default='/GPFS/data/yaxindu/FedHomo/dataset', type=str)
    parser.add_argument('--VHL_dataset_batch_size', default=128, type=int)
    parser.add_argument('--VHL_dataset_list', default="Gaussian_Noise", type=str, help="either Gaussian_Noise or style_GAN_init")
    parser.add_argument('--VHL_align_local_epoch', default=5, type=int)

    # FedGen
    parser.add_argument("--gen_noise_dim", type=int, default=512)
    parser.add_argument("--gen_generator_learning_rate", type=float, default=0.005)
    parser.add_argument("--gen_hidden_dim", type=int, default=512)
    parser.add_argument("--gen_server_epochs", type=int, default=400)
    parser.add_argument("--gen_batch_size", type=int, default=32)
    
    args = parser.parse_args()
    
    dataset_info = {
        'mnist': {'n_class': 10, 'n_channel': 1, 'img_size': '28'},
        'fashionmnist': {'n_classes': 10, 'n_channel': 1, 'img_size': '28'},
        'femnist': {'n_class': 62, 'n_channel': 1, 'img_size': '28'},
        'cifar10': {'n_class': 10, 'n_channel': 3, 'img_size': '32'},
        'cifar100': {'n_class': 100, 'n_channel': 3, 'img_size': '32'},
    }

    args.n_class = dataset_info[args.dataset]['n_class']
    args.n_channel = dataset_info[args.dataset]['n_channel']
    args.img_size = dataset_info[args.dataset]['img_size']
    
    if args.dataset == 'femnist':
        args.n_parties = args.femnist_train_num

    if args.dataset == 'flair':
        args.img_size = args.flair_img_size

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print_parsed_args(args)

    return args

def print_parsed_args(args):
    print(f"{'='*20} Parsed Arguments {'='*20}")  
  
    # Determine the maximum length of argument names  
    max_key_length = max(len(key) for key in vars(args).keys())  
  
    # Iterate through the parsed arguments and print them with proper padding  
    for key, value in vars(args).items():  
        print(f"{key.ljust(max_key_length)}: {value}")  
  
    print('=' * 40)