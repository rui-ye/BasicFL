import torch
import numpy as np
import random

def set_seed(seed):  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    random.seed(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 

def set_model_zero(model):
    model_state = model.state_dict()
    for key in model_state.keys():
        model_state[key] = torch.zeros_like(model_state[key])
    model.load_state_dict(model_state)
    return model