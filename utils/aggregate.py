

def aggregate_model(global_model, local_model_list, client_num_samples, available_clients):
    total_data_points = sum([client_num_samples[r] for r in available_clients])
    fed_avg_freqs = [client_num_samples[r] / total_data_points for r in available_clients]
    
    global_w = global_model.state_dict()
    for net_id, client_id in enumerate(available_clients):
        net_para = local_model_list[client_id].state_dict()
        if net_id == 0:
            for key in net_para:
                global_w[key] = net_para[key] * fed_avg_freqs[net_id]
        else:
            for key in net_para:
                global_w[key] += net_para[key] * fed_avg_freqs[net_id]
    global_model.load_state_dict(global_w)