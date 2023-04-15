import os
import sys
import argparse
import numpy as np
import time
import json
import SimpleITK as sitk
from einops import rearrange
from tqdm import tqdm
from datetime import timedelta
import skimage.morphology as morphology

import torch
import torch.nn as nn
import torch.utils.data as tudata
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.distributed as distrib
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler

from models import unet, srs, trans_unet
from utils import util
from utils.image_util import unnormalize_normal
from datasets.generic_dataset import *

from path_util import *
from file_io import *
from evaluation.evaluation import NonOverlapCropEvaluation, MostFitCropEvaluation, OverlapCropEvaluation

parser = argparse.ArgumentParser(
    description='Super Resolution Segmentation')
# data specific
parser.add_argument('--data_file', default='/PBshare/SEU-ALLEN/Users/Gaoyu/neuronSegSR/Task501_neuron/data_splits.pkl',
                    type=str, help='dataset split file')
# training specific
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--image_shape', default='32,64,64', type=str,
                    help='Input image shape')
parser.add_argument('--res_rescale', default='4,1,1', type=str,
                    help='Super resolution scale')                    
parser.add_argument('--cpu', action="store_true",
                    help='Whether use gpu to train model, default True')
parser.add_argument('--loss_weight', default='1,5',
                    help='The weight of loss_ce and loss_box')
parser.add_argument('--amp', action="store_true",
                    help='Whether to use AMP training, default True')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.99, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=0, type=float,
                    help='Weight decay')
parser.add_argument('--decay_type', choices=['ploy', 'linear'], default='ploy', type=str,
                    help='How to decay the learning rate')
parser.add_argument('--warmup_steps', default=100, type=str,
                    help='Step of training to perform learning rate warmup for')
parser.add_argument('--max_grad_norm', default=1.0, type=float,
                    help='Max gradient norm.')
parser.add_argument('--max_epochs', default=200, type=int,
                    help='maximal number of epochs')
parser.add_argument('--gce_q', default=0.4, type=float,
                    help='q in gce loss')
parser.add_argument('--step_per_epoch', default=200, type=int,
                    help='step per epoch')
parser.add_argument('--deterministic', action='store_true',
                    help='run in deterministic mode')
parser.add_argument('--test_frequency', default=3, type=int,
                    help='frequency of testing')
parser.add_argument('--print_frequency', default=5, type=int,
                    help='frequency of information logging')
parser.add_argument('--local_rank', default=-1, type=int, metavar='N',
                    help='Local process rank')  # DDP required
parser.add_argument('--seed', default=1025, type=int,
                    help='Random seed value')
parser.add_argument('--checkpoint', default='', type=str,
                    help='Saved checkpoint')
parser.add_argument('--evaluation', action='store_true',
                    help='evaluation')
parser.add_argument('--phase', default='train')

# network specific
parser.add_argument('--net_config', default="./models/configs/default_config.json",
                    type=str,
                    help='json file defining the network configurations')

parser.add_argument('--save_folder', default='exps/temp',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


def ddp_print(content):
    if args.is_master:
        print(content)


def save_image_in_training(imgfiles, img, lab, logits, epoch, phase, idx):    
    # the shape of image [batch_size, c, z, y, x] 
    imgfile = imgfiles[idx]
    prefix = get_file_prefix(imgfile)
    with torch.no_grad():
        img_v = (unnormalize_normal(img[idx].numpy())[0]).astype(np.uint8)
        lab_v = (unnormalize_normal(lab[idx].numpy().astype(np.float32))[0]).astype(np.uint8)
        
        logits = F.softmax(logits, dim=1).to(torch.device('cpu'))
        log_v = (unnormalize_normal(logits[idx,[1]].numpy())[0]).astype(np.uint8)

        if phase == 'train':
            out_img_file = f'debug_epoch{epoch}_{prefix}_{phase}_img.v3draw'
            out_lab_file = f'debug_epoch{epoch}_{prefix}_{phase}_lab.v3draw'
            out_pred_file = f'debug_epoch{epoch}_{prefix}_{phase}_pred.v3draw'
        elif phase == 'val':
            out_img_file = f'debug_epoch{epoch}_{prefix}_{phase}_img.v3draw'
            out_lab_file = f'debug_epoch{epoch}_{prefix}_{phase}_lab.v3draw'
            out_pred_file = f'debug_epoch{epoch}_{prefix}_{phase}_pred.v3draw'
        elif phase == 'test':
            out_img_file = f'{prefix}_{phase}_img.v3draw'
            out_lab_file = f'{prefix}_{phase}_lab.v3draw'
            out_pred_file = f'{prefix}_{phase}_pred.v3draw'


        save_image(os.path.join(args.save_folder, out_img_file), img_v[np.newaxis, :])
        save_image(os.path.join(args.save_folder, out_lab_file), lab_v[np.newaxis, :])
        save_image(os.path.join(args.save_folder, out_pred_file), log_v[np.newaxis, :])


def get_fn_weights(lab_d, probs, bg_thresh=0.5, weight_fn=5.0, start_epoch=5):
    if args.curr_epoch < start_epoch:
        loss_weights, loss_weights_unsq = 1.0, 1.0
    else:
        # foreground
        pos_mask = lab_d > 0
        # background predicted
        bg_mask = probs[:,0] > bg_thresh
        fn_mask = pos_mask & bg_mask
        loss_weights = torch.ones(fn_mask.size(), dtype=probs.dtype, device=probs.device)
        loss_weights[fn_mask] = weight_fn
        loss_weights_unsq = loss_weights.unsqueeze(1)
    return loss_weights, loss_weights_unsq


def get_forward(img, lab, crit_ce, crit_dice, model):
    # img: b, c, z, y, x
    # lab: b, z, y, x
    # logits: b, c, z, y, x
    logits = model(img)
    probs = F.softmax(logits, dim=1)

        # hard positive mining. NOTE: we can only do positive mining, as the label is incomplete
    do_hard_pos_mining = True
    if do_hard_pos_mining:
        loss_weights, loss_weights_unsq = get_fn_weights(lab, probs, bg_thresh=0.5, weight_fn=3.0, start_epoch=80)
    else:
        loss_weights = 1.0
        loss_weights_unsq = 1.0
    
    loss_ce = (crit_ce(logits, lab.long()) * loss_weights).mean()
    loss_dice = crit_dice(logits * loss_weights_unsq, lab.float() * loss_weights)
    loss = loss_ce + loss_dice

    return loss_ce, loss_dice, loss, probs


def get_forward_eval(img, lab, crit_ce, crit_dice, model):
    if args.amp:
        with autocast():
            with torch.no_grad():
                loss_ce, loss_dice, loss, pred = get_forward(img, lab, crit_ce, crit_dice, model)
    else:
        with torch.no_grad():
            loss_ce, loss_dice, loss, pred = get_forward(img, lab, crit_ce, crit_dice, model)
    return loss_ce, loss_dice, loss, pred


def validate(model, val_loader, crit_ce, crit_dice, epoch, debug=True, num_image_save=10, phase='val'):
    model.eval()
    num_saved = 0
    if num_image_save == -1:
        num_image_save = 9999

    losses = []
    processed = -1
    for img, lab, imgfiles, swcfiles in val_loader:
        processed += 1

        img_d = img.to(args.device)
        lab_d = lab.to(args.device)
        if phase == 'val':
            loss_ce, loss_dice, loss, pred = get_forward_eval(img_d, lab_d, crit_ce, crit_dice, model)

        elif phase == 'test' or phase == 'par':
            ddp_print(f'==> processed: {processed} current:{imgfiles}')
            noce = OverlapCropEvaluation(args.imgshape)
            assert args.batch_size == 1, "Batch size must be 1 for test phase for current version"

            n_ens = 1
            for ie in range(n_ens):
                if ie == 0:
                    crops, crop_sizes, lab_crops,crop_lab_sizes = noce.get_image_crops(img_d[0], lab_d[0])
                elif ie == 1:
                    crops, crop_sizes, lab_crops = noce.get_image_crops(torch.flip(img_d[0], [2]), torch.flip(lab_d[0], [1]))
                elif ie == 2:
                    crops, crop_sizes, lab_crops = noce.get_image_crops(torch.flip(img_d[0], [3]), torch.flip(lab_d[0], [2]))
                elif ie == 3:
                    crops, crop_sizes, lab_crops = noce.get_image_crops(torch.flip(img_d[0], [2,3]), torch.flip(lab_d[0], [1,2]))

                logits_list = []
                loss_ce, loss_dice, loss = [], [], 0
                for i in range(len(crops)):
                    loss_ces_i, loss_dices_i, loss_i, logits_i = get_forward_eval(crops[i][None], lab_crops[i][None], crit_ce, crit_dice, model)
                    logits_list.append(logits_i[0])
                    loss_ce.append(loss_ces_i.cpu())
                    loss_dice.append(loss_dices_i.cpu())
                    loss += loss_i
                

                # merge the crop of prediction to unit one
                logits = noce.get_pred_from_crops(lab_d[0].shape, logits_list, crop_lab_sizes)[None]

                if ie == 0:
                    avg_logits = logits
                elif ie == 1:
                    avg_logits += torch.flip(logits, [3])
                elif ie == 2:
                    avg_logits += torch.flip(logits, [4])
                elif ie == 3:
                    avg_logits += torch.flip(logits, [3,4])
                else:
                    raise ValueError

                ncrop = len(crops)
                del crops, lab_crops, logits_list

                # average the loss
                loss_ce = np.array(loss_ce).mean(axis=0)
                loss_dice = np.array(loss_dice).mean(axis=0)
                loss /= ncrop
                #TODO: the loss for ensemble mode should also averaged
                
            # averaging all logits
            pred = avg_logits / n_ens

        else:
            raise ValueError

        del img_d
        del lab_d


        losses.append([loss_ce.item(), loss_dice.item(), loss.item()])

        if debug:
            for debug_idx in range(img.size(0)):
                num_saved += 1
                if num_saved > num_image_save:
                    break
                save_image_in_training(imgfiles, img, lab, pred, epoch, phase, debug_idx)

    losses = torch.from_numpy(np.array(losses)).to(args.device).mean(dim=0)

    return losses[0], losses[1], losses[2]


def load_dataset(phase, imgshape):
    dset = GenericDataset(args.data_file, phase=phase, imgshape=imgshape, res_rescale=args.res_rescale)
    ddp_print(f'Number of {phase} samples: {len(dset)}')
    # distributedSampler
    if phase == 'train':
        sampler = RandomSampler(dset) if args.local_rank == -1 else DistributedSampler(dset, shuffle=True)
    else:
        sampler = RandomSampler(dset) if args.local_rank == -1 else DistributedSampler(dset, shuffle=False)

    loader = tudata.DataLoader(dset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=False, pin_memory=True,
                               sampler=sampler,
                               drop_last=True,
                               worker_init_fn=util.worker_init_fn)
    dset_iter = iter(loader)
    return loader, dset_iter


def evaluate(model, optimizer, crit_ce, crit_dice, imgshape, phase):
    val_loader, val_iter = load_dataset(phase, imgshape)
    args.curr_epoch = 0
    loss_ce, loss_dice, *_ = validate(model, val_loader, crit_ce, crit_dice, epoch=0, debug=True, num_image_save=-1,
                                        phase=phase)
    ddp_print(f'Average loss_ce and loss_dice: {loss_ce:.5f} {loss_dice:.5f}')


def train(model, optimizer, crit_ce, crit_dice, imgshape):
    # dataset preparing
    train_loader, train_iter = load_dataset('train', imgshape)
    val_loader, val_iter = load_dataset('val', imgshape)
    args.step_per_epoch = len(train_loader) if len(train_loader) < args.step_per_epoch else args.step_per_epoch
    t_total = args.max_epochs * args.step_per_epoch
    if args.decay_type == "ploy":
        scheduler = util.PolynomialLR(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = util.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # training process
    model.train()
    t0 = time.time()
    # for automatic mixed precision
    grad_scaler = GradScaler()
    debug = True
    debug_idx = 0
    for epoch in range(args.max_epochs):
        # push the epoch information to global namespace args
        args.curr_epoch = epoch

        avg_loss_ce = 0
        avg_loss_dice = 0

        epoch_iterator = tqdm(train_loader,
                        desc=f'Epoch {epoch + 1}/{args.max_epochs}',
                        total=args.step_per_epoch,
                        postfix=dict,
                        dynamic_ncols=True,
                        disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            img, lab, imgfiles, swcfiles = batch

            img_d = img.to(args.device)
            lab_d = lab.to(args.device)

            optimizer.zero_grad()
            if args.amp:
                with autocast():
                    loss_ce, loss_dice, loss, pred = get_forward(img_d, lab_d, crit_ce, crit_dice, model)
                    del img_d
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()
            else:
                loss_ce, loss_dice, loss = get_forward(img_d, lab_d, crit_ce, crit_dice, model)
                del img_d
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

            avg_loss_ce += loss_ce
            avg_loss_dice += loss_dice

            # train statistics for bebug afterward
            # if step % args.print_frequency == 0:
            #     ddp_print(
            #         f'[{epoch}/{step}] loss_ce={loss_ce:.5f}, loss_box={loss_box:.5f}, accuracy_cls={accuracy_cls:.3f}, accuracy_pos={accuracy_pos:.3f}, time: {time.time() - t0:.4f}s')

            epoch_iterator.set_postfix({'loss_ce': loss_ce.item(), 'loss_dice': loss_dice.item()})

        avg_loss_ce /= args.step_per_epoch
        avg_loss_dice /= args.step_per_epoch

        # do validation
        if epoch % args.test_frequency == 0:
            ddp_print('Evaluate on val set')
            val_loss_ce, val_loss_dice, val_loss = validate(model, val_loader, crit_ce, crit_dice, epoch, debug=debug,
                                                            phase='val')

            model.train()  # back to train phase
            ddp_print(f'[Val{epoch}] average ce loss, dice loss and the sum are {val_loss_ce:.5f}, {val_loss_dice:.5f}, {val_loss:.5f}')
            # save the model
            if args.is_master:
                # save current model
                torch.save(model, os.path.join(args.save_folder, 'final_model.pt'))

        # save image for subsequent analysis
        if debug and args.is_master and epoch % args.test_frequency == 0:
            save_image_in_training(imgfiles, img, lab, pred, epoch, 'train', debug_idx)


def main():
    # keep track of master, useful for IO
    args.is_master = args.local_rank in [0, -1]

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    if args.deterministic:
        util.set_deterministic(deterministic=True, seed=args.seed)

    # for output folder
    if args.is_master and not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    # Network
    with open(args.net_config) as fp:
        net_configs = json.load(fp)
        print('Network configs: ', net_configs)
        model = unet.UNet(**net_configs)
        ddp_print('\n' + '=' * 10 + 'Network Structure' + '=' * 10)
        ddp_print(model)
        ddp_print('=' * 30 + '\n')

    model = model.to(args.device)
    if args.checkpoint:
        # load checkpoint
        ddp_print(f'Loading checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location='cuda:0')
        model.load_state_dict(checkpoint.state_dict())
        del checkpoint
        # if args.is_master:
        #    torch.save(checkpoint.module.state_dict(), "exp040.state_dict")
        #    sys.exit()

    # convert to distributed data parallel model
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank],
                output_device=args.local_rank)  # , find_unused_parameters=True)

    # optimizer & loss
    if args.checkpoint:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, amsgrad=True)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)

    # crit_ce = util.GCELoss(q=args.gce_q).to(args.device)
    crit_ce = nn.CrossEntropyLoss(reduction='none').to(args.device)
    crit_dice = util.BinaryDiceLoss(smooth=1e-5, input_logits=True).to(args.device)
    args.imgshape = tuple(map(int, args.image_shape.split(',')))
    args.res_rescale = tuple(map(int, args.res_rescale.split(',')))
    # loss_weight = list(map(float, args.loss_weight.split(',')))


    # Print out the arguments information
    ddp_print('Argument are: ')
    ddp_print(f'   {args}')

    if args.evaluation:
        evaluate(model, optimizer, crit_ce, crit_dice, args.imgshape, args.phase)
    else:
        train(model, optimizer, crit_ce, crit_dice, args.imgshape)


if __name__ == '__main__':
    main()
