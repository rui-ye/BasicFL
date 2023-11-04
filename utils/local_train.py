import torch
import copy

def local_train_base(args, train_dls, local_model_list, global_model, available_clients):
    
    # Sync model parameters
    global_w = global_model.state_dict()
    for model in local_model_list:
        model.load_state_dict(global_w)
    
    # Conduct local model training on available clients
    for client_id in available_clients:
        model = local_model_list[client_id].cuda()
        train_loader = train_dls[client_id]
        optimizer = optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        model.train()

        iterator = iter(train_loader)
        for iteration in range(args.n_iteration):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                x, target = next(iterator)
            x, target = x.cuda(), target.long().cuda()

            optimizer.zero_grad()

            out = model(x)

            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
        model.to('cpu')

def local_train_fedprox(args, train_dls, local_model_list, global_model, available_clients):
    
    # Sync model parameters
    global_w = global_model.state_dict()
    for model in local_model_list:
        model.load_state_dict(global_w)
    global_model.cuda()
    
    # Conduct local model training on available clients
    for client_id in available_clients:
        model = local_model_list[client_id].cuda()
        train_loader = train_dls[client_id]
        optimizer = optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        model.train()

        iterator = iter(train_loader)
        for iteration in range(args.n_iteration):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                x, target = next(iterator)
            x, target = x.cuda(), target.long().cuda()
            optimizer.zero_grad()
            
            out = model(x)

            loss = criterion(out, target)
            for param_p, param in zip(model.parameters(), global_model.parameters()):
                loss += ((args.mu / 2) * torch.norm((param - param_p)) ** 2)
                
            loss.backward()
            optimizer.step()
        model.to('cpu')
    global_model.to('cpu')

def local_train_scaffold(args, train_dls, local_model_list, global_model, available_clients, local_auxiliary_list, global_auxiliary):
    
    # Sync model parameters
    global_w = global_model.state_dict()
    for model in local_model_list:
        model.load_state_dict(global_w)
    
    # SCAFFOLD delta
    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    
    global_auxiliary.cuda()
    global_model.cuda()
    
    # Conduct local model training on available clients
    for client_id in available_clients:
        model = local_model_list[client_id].cuda()
        train_loader = train_dls[client_id]

        auxiliary_model = local_auxiliary_list[client_id].cuda()
        auxiliary_global_para = global_auxiliary.state_dict()
        auxiliary_model_para = auxiliary_model.state_dict()

        optimizer = optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()

        iterator = iter(train_loader)
        for iteration in range(args.n_iteration):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                x, target = next(iterator)
            x, target = x.cuda(), target.long().cuda()

            optimizer.zero_grad()

            out = model(x)

            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            net_para = model.state_dict()
            for key in net_para:
                net_para[key] = net_para[key] - args.lr * (auxiliary_global_para[key] - auxiliary_model_para[key])
            model.load_state_dict(net_para)
        
        auxiliary_new_para = auxiliary_model.state_dict()
        auxiliary_delta_para = copy.deepcopy(auxiliary_model.state_dict())
        global_model_para = global_model.state_dict()
        net_para = model.state_dict()
        for key in net_para:
            auxiliary_new_para[key] = auxiliary_new_para[key] - auxiliary_global_para[key] + (global_model_para[key] - net_para[key]) / (args.n_iteration * args.lr)
            auxiliary_delta_para[key] = auxiliary_new_para[key] - auxiliary_model_para[key]
        auxiliary_model.load_state_dict(auxiliary_new_para)
        auxiliary_model.to('cpu')

        for key in total_delta:
            total_delta[key] += auxiliary_delta_para[key]

        model.to('cpu')
    global_model.to('cpu')

    for key in total_delta:
        total_delta[key] /= args.n_client
    auxiliary_global_para = global_auxiliary.state_dict()
    for key in auxiliary_global_para:
        if auxiliary_global_para[key].type() == 'torch.LongTensor':
            auxiliary_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif auxiliary_global_para[key].type() == 'torch.cuda.LongTensor':
            auxiliary_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            auxiliary_global_para[key] += total_delta[key]
    global_auxiliary.load_state_dict(auxiliary_global_para)