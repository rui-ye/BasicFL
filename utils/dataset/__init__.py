from .dataset_torch import cifar_dataset_read, fashionmnist_dataset_read
from .dataset_leaf import leaf_read

def get_dataloader(args):
    if args.dataset in ('cifar10', 'cifar100'):
        train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions = cifar_dataset_read(args, args.dataset, args.datadir, args.batch_size, args.n_client, args.partition, args.beta, args.skew_class)
    elif args.dataset == 'fashionmnist':
        train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions = fashionmnist_dataset_read(args, args.dataset, args.datadir, args.batch_size, args.n_client, args.partition, args.beta, args.skew_class)
    elif args.dataset in ('femnist'):
        train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions = leaf_read(args)
    return train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions