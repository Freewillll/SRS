
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, p=1, reduction='mean', input_logits=True):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.input_logits = input_logits

    def forward(self, logits, gt_float):
        assert logits.shape[0] == gt_float.shape[0], "batch size error!"
        if self.input_logits:
            probs = F.softmax(logits, dim=1)[:,1]    # foreground
        else:
            probs = logits[:,1]        

        probs = probs.contiguous().view(probs.shape[0], -1)
        gt_float = gt_float.contiguous().view(gt_float.shape[0], -1)

        nominator = 2 * torch.sum(torch.mul(probs, gt_float), dim=1) + self.smooth
        if self.p == 1:
            denominator = torch.sum(probs + gt_float, dim=1) + self.smooth
        elif self.p == 2:
            denominator = torch.sum(probs*probs + gt_float*gt_float, dim=1) + self.smooth
        else:
            raise NotImplementedError

        loss = 1 - nominator / denominator
        return loss.mean()


class TruncatedLoss(nn.Module):

    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
             
    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)
        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)


class GCELoss(nn.Module):
    def __init__(self, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q
    
    def forward(self, logits, lab):
        if self.q == 0.0:
            return nn.CrossEntropyLoss
        else:
            pred = F.softmax(logits, dim=1)
            p = torch.gather(pred, 1, torch.unsqueeze(lab, 1))
            loss = (1- (p**self.q)) / self.q
        return loss


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


# class PloySchedule(LambdaLR):
#     def __init__(self, optimizer, t_total, last_epoch=-1):
#         self.t_total = t_total
#         super(PloySchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

#     def lr_lambda(self, step):
#         return (1 - step / self.t_total)**0.9
    

class PolynomialLR(LambdaLR):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        t_total,
        power=0.9,
        last_epoch=-1,
    ):
        self.warmup_steps = int(warmup_steps)
        self.t_total = int(t_total)
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return step / self.warmup_steps * (1 - self.warmup_steps / self.t_total) ** self.power
        else:
            return (1 - step / self.t_total) ** self.power


# def compute_metric(logits, lab, num_classes, reduce_zero_label):
#     # logits: b, c, z, y, x
#     # lab: b, z, y, x
    


def init_device(device_name):
    if type(device_name) == int:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            raise EnvironmentError("GPU is not accessible!")
    elif type(device_name) == str:
        if device_name == 'cpu':
            device = torch.device(device_name)
        elif device_name[:4] == 'cuda':
            if torch.cuda.is_available():
                device = torch.device(device_name)
        else:
            raise ValueError("Invalid name for device")
    else:
        raise NotImplementedError
    return device


def set_deterministic(deterministic=True, seed=1024):
    if deterministic:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return True


def worker_init_fn(worker_id):
    """Function to avoid numpy.random seed duplication across multi-threads"""
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def logits_to_seg(logits, thresh=None):
    with torch.no_grad():
        if thresh is None:
            # no need to do expensive softmax
            seg = logits.argmax(dim=1)
        else:
            probs = F.softmax(logits, dim=1)
            vmax, seg = probs.max(dim=1)
            mask = vmax > thresh
            # thresh for non-zero class
            seg[~mask] = 0
    return seg


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    """ 
    poly_lr policy as the same as nnUNet
    """
    return initial_lr * (1 - epoch / max_epochs)**exponent


def step_lr(epoch, steps, initial_lr, scale_factor=0.2):
    lr = initial_lr
    for step in steps:
        if epoch > step:
            lr *= scale_factor
        else:
            break
    return lr

# TODO: network ploting
