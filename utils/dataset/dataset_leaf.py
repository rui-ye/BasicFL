import os
from collections import defaultdict, OrderedDict
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def read_json_files(data_dir, files):
    clients = []
    data = defaultdict(lambda: None)
    # files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        data.update(cdata['user_data'])

    clients = list(data.keys())
    return clients, data

class FEMNIST():
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    def __init__(self, datadir, train_user_index, test_user_index, train=True, transform=None, target_transform=None):
        super(FEMNIST, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        # train_clients, train_groups, train_data_temp, test_data_temp = read_data("./data/femnist/train",
        #                                                                          "./data/femnist/test")
        files = os.listdir(datadir)
        files = [f for f in files if f.endswith('.json')]
        
        clients, data = read_json_files(datadir, files)
        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for user in train_user_index:
                self.dic_users[user] = set()
                l = len(train_data_x)
                cur_x = data[clients[user]]['x']
                cur_y = data[clients[user]]['y']
                for j in range(len(cur_x)):
                    self.dic_users[user].add(j + l)
                    train_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.target = train_data_y
        else:  # test就混杂在一起搞
            test_data_x = []
            test_data_y = []
            for user in test_user_index:
                cur_x = data[clients[user]]['x']
                cur_y = data[clients[user]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    test_data_y.append(cur_y[j])

            self.data = test_data_x
            self.target = test_data_y

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        img = np.array([img])
        # img = Image.fromarray(img, mode='L')
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return torch.from_numpy((0.5-img)/0.5).float(), target

    def __len__(self):
        return len(self.data)
    
    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def split_leaf_data(args):
    # function to split dataset LEAF
    datadir = '/GPFS/data/jingyichai/datasets/LEAF/femnist/data/all_data'
    
    files = os.listdir(datadir)
    files = [f for f in files if f.endswith('.json')]
    print(files)
    rng = random.Random(args.init_seed)

    # check if data contains information on hierarchies
    file_dir = os.path.join(datadir, files[0])
    with open(file_dir, 'r') as inf:
        data = json.load(inf)
    include_hierarchy = 'hierarchies' in data
    print("include hierarchy? ", include_hierarchy)

    usrs_samples_num = []
    usrs_num = []
    users = []
    for f in files:
        file_dir = os.path.join(datadir, f)
        with open(file_dir, 'r') as inf:
            # Load data into an OrderedDict, to prevent ordering changes
            # and enable reproducibility
            data = json.load(inf, object_pairs_hook=OrderedDict)
            usrs_num.append(len(data['users']))
            users.extend(data['users'])
            usrs_samples_num.extend(data['num_samples'])
    if args.femnist_sample_top:
        indices = sorted(range(len(usrs_samples_num)), key = lambda x:usrs_samples_num[x], reverse=True)
    else:
        indices = list(range(len(usrs_samples_num)))
    num_users = args.femnist_train_num + args.femnist_test_num
    indices = indices[:num_users]
    print("indices: ",indices)
    rng.shuffle(indices)
    train_indices = indices[:args.femnist_train_num]  # obtain the train index
    test_indices = indices[args.femnist_train_num:]  # obtain the test index

    return train_indices, test_indices, users  # 没有在train_user_index里面的就是test数据

def leaf_read(args):
    datadir = '/GPFS/data/jingyichai/datasets/LEAF/femnist/data/all_data'
    train_user_index, test_user_index, users = split_leaf_data(args)
    if args.dataset == 'femnist':
        test_ds = FEMNIST(datadir, train_user_index, test_user_index, train=False)
        test_dl = DataLoader(dataset=test_ds, batch_size=32, shuffle=True)
        train_ds = FEMNIST(datadir, train_user_index, test_user_index, train=True)
        print(f'number of training / testing samples: {len(train_ds)} / {len(test_ds)}')
    train_dataloaders = []
    client_num_samples = []
    dict_users = train_ds.get_client_dic()
    num_classes = np.unique(train_ds.target).shape[0]
    print("num_classes: ", num_classes)
    traindata_cls_counts = []
    for net_id in train_user_index: 
        idxs = dict_users[net_id]
        client_num_samples.append(len(idxs))
        train_ds_local = DatasetSplit(train_ds, idxs)
        train_dl_local = DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True, drop_last=True)
        train_dataloaders.append(train_dl_local)
        traindata_cls_counts.append(count_class(train_ds, num_classes, idxs))
    traindata_cls_counts = np.array(traindata_cls_counts)
    data_distributions = traindata_cls_counts / traindata_cls_counts.sum(axis=1)[:,np.newaxis]

    return train_dataloaders, test_dl, client_num_samples, traindata_cls_counts, data_distributions

def count_class(train_ds, num_classes, idxs):
    targets = train_ds.target
    target = [targets[i] for i in idxs]
    traindata_cls_count_per_client = np.zeros(num_classes)
    for t in target:
        traindata_cls_count_per_client[t] += 1
    return traindata_cls_count_per_client