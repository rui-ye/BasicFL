import torch

def evaluation(args, global_model, test_dl, best_acc):

    acc = compute_acc(global_model, test_dl)
    best_acc = acc if best_acc < acc else best_acc
    return acc, best_acc

def compute_acc(net, test_data_loader):
    net.cuda()
    net.eval()
    top1, top2, top5, total = 0, 0, 0, 0
    if isinstance(test_data_loader, list):
        with torch.no_grad():
            for test_dl in  test_data_loader:
                for batch_idx, (x, target) in enumerate(test_dl):
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                    out = net(x)
                    _, pred_label = torch.topk(out.data, 5, 1)
                    total += x.data.size()[0]
                    target = target.reshape(-1, 1)
                    top1 += (pred_label[:,0:1] == target.data).sum().item()
                    top2 += (pred_label[:,0:2] == target.data).sum().item()
                    top5 += (pred_label == target.data).sum().item()
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data_loader):
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                out = net(x)
                _, pred_label = torch.topk(out.data, 5, 1)
                total += x.data.size()[0]
                target = target.reshape(-1, 1)
                top1 += (pred_label[:,0:1] == target.data).sum().item()
                top2 += (pred_label[:,0:2] == target.data).sum().item()
                top5 += (pred_label == target.data).sum().item()
    net.to('cpu')
    return top1 / float(total)