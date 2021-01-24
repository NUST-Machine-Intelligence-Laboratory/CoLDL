import os
import pathlib
import time
import datetime
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from apex import amp
from utils.core import accuracy, evaluate
from utils.builder import *
from utils.utils import *
from utils.meter import AverageMeter
from utils.logger import Logger, print_to_logfile, step_flagging
from utils.plotter import plot_results_cotraining
from utils.loss import cross_entropy, entropy_loss, regression_loss
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CLDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        return x1, x2


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_scale, projection_size, init_method='He'):
        super().__init__()

        mlp_hidden_size = round(mlp_hidden_scale * in_channels)
        self.mlp_head = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            # nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )
        init_weights(self.mlp_head, init_method)

    def forward(self, x):
        return self.mlp_head(x)


class ResNet(nn.Module):
    def __init__(self, arch='resnet18', num_classes=200, pretrained=True):
        super().__init__()
        assert arch in torchvision.models.__dict__.keys(), f'{arch} is not supported!'
        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.feat_dim = resnet.fc.in_features
        self.neck = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        mlp_scale = 1
        self.projector_head = MLPHead(in_channels=self.feat_dim, mlp_hidden_scale=mlp_scale, projection_size=num_classes)
        self.predictor_head = nn.Linear(in_features=num_classes, out_features=num_classes)
        init_weights(self.predictor_head, init_method='He')

    def forward(self, x):
        N = x.size(0)
        x = self.backbone(x)
        x = self.neck(x).view(N, -1)
        embedding = self.projector_head(x)
        prediction = self.predictor_head(embedding)

        return {'logits': prediction, 'embeddings': embedding}


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def make_lr_plan(init_lr, stage1, epochs):
    lrs = [init_lr] * epochs

    init_lr_stage_ldl = init_lr
    for t in range(stage1, epochs):
        lrs[t] = 0.5 * init_lr_stage_ldl * (1 + math.cos((t - stage1 + 1) * math.pi / (epochs - stage1 + 1)))

    return lrs


def get_smoothed_label_distribution(labels, nc, epsilon):
    smoothed_label = torch.full(size=(labels.size(0), nc), fill_value=epsilon / (nc - 1))
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1).cpu(), value=1 - epsilon)
    return smoothed_label.to(labels.device)


def sample_selector(l1, l2, drop_rate):
    ind_sorted_1 = torch.argsort(l1.data)  # ascending order
    ind_sorted_2 = torch.argsort(l2.data)
    num_remember = max(int((1 - drop_rate) * l1.shape[0]), 1)
    ind_clean_1 = ind_sorted_1[:num_remember]
    ind_clean_2 = ind_sorted_2[:num_remember]
    ind_unclean_1 = ind_sorted_1[num_remember:]
    ind_unclean_2 = ind_sorted_2[num_remember:]
    return {'clean1': ind_clean_1, 'clean2': ind_clean_2, 'unclean1': ind_unclean_1, 'unclean2': ind_unclean_2}


def main(cfg, device):
    init_seeds()
    cfg.use_fp16 = False if device.type == 'cpu' else cfg.use_fp16

    # logging ----------------------------------------------------------------------------------------------------------------------------------------
    logger_root = f'Results/{cfg.dataset}'
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(logger_root, f'{logtime}-{cfg.log}')
    # result_dir = os.path.join(logger_root, f'ablation_study-{cfg.log}')  #TODO
    logger = Logger(logging_dir=result_dir, DEBUG=False)
    logger.set_logfile(logfile_name='log.txt')
    save_params(cfg, f'{result_dir}/params.json', json_format=True)
    logger.debug(f'Result Path: {result_dir}')

    # model, optimizer, scheduler --------------------------------------------------------------------------------------------------------------------
    opt_lvl = 'O1' if cfg.use_fp16 else 'O0'
    n_classes = cfg.n_classes
    net1 = ResNet(arch=cfg.net1, num_classes=n_classes, pretrained=True)
    optimizer1 = build_sgd_optimizer(net1.parameters(), cfg.lr, cfg.weight_decay)
    net1, optimizer1 = amp.initialize(net1.to(device), optimizer1, opt_level=opt_lvl, keep_batchnorm_fp32=None, loss_scale=None, verbosity=0)
    net2 = ResNet(arch=cfg.net2, num_classes=n_classes, pretrained=True)
    optimizer2 = build_sgd_optimizer(net2.parameters(), cfg.lr, cfg.weight_decay)
    net2, optimizer2 = amp.initialize(net2.to(device), optimizer2, opt_level=opt_lvl, keep_batchnorm_fp32=None, loss_scale=None, verbosity=0)
    lr_plan = make_lr_plan(cfg.lr, cfg.stage1, cfg.epochs)

    with open(f'{result_dir}/network.txt', 'w') as f:
        f.writelines(net1.__repr__())
        f.write('\n\n---------------------------\n\n')
        f.writelines(net1.__repr__())

    # drop rate scheduler ----------------------------------------------------------------------------------------------------------------------------
    T_k = cfg.stage1
    final_drop_rate = 0.25
    final_ldl_rate = cfg.ldl_rate
    drop_rate_scheduler = np.ones(cfg.epochs) * final_drop_rate
    drop_rate_scheduler[:T_k] = np.linspace(0, final_drop_rate, T_k)
    drop_rate_scheduler[T_k:cfg.epochs] = np.linspace(final_drop_rate, final_ldl_rate, cfg.epochs - T_k)

    # dataset, dataloader ----------------------------------------------------------------------------------------------------------------------------
    transform = build_transform(rescale_size=cfg.rescale_size, crop_size=cfg.crop_size)
    dataset = build_webfg_dataset(os.path.join(cfg.database, cfg.dataset), CLDataTransform(transform['train']), transform['test'])
    logger.debug(f"Number of Training Samples: {dataset['n_train_samples']}")
    logger.debug(f"Number of Testing  Samples: {dataset['n_test_samples']}")
    train_loader = DataLoader(dataset['train'], batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    # meters -----------------------------------------------------------------------------------------------------------------------------------------
    train_loss1, train_loss2 = AverageMeter(), AverageMeter()
    train_accuracy1, train_accuracy2 = AverageMeter(), AverageMeter()
    iter_time = AverageMeter()

    # training ---------------------------------------------------------------------------------------------------------------------------------------
    start_epoch = 0
    best_accuracy1, best_accuracy2 = 0.0, 0.0
    best_epoch1, best_epoch2 = None, None

    if cfg.dataset == 'cifar100' and cfg.noise_type != 'clean':
        t = torch.tensor(dataset['train'].noisy_labels)
    else:
        t = torch.tensor(dataset['train'].targets)
    labels2learn1 = torch.full(size=(dataset['n_train_samples'], n_classes), fill_value=0.0)
    labels2learn1.scatter_(dim=1, index=torch.unsqueeze(t, dim=1), value=1.0 * 10)
    labels2learn2 = labels2learn1

    flag = [0, 0, 0]
    for epoch in range(start_epoch, cfg.epochs):
        start_time = time.time()
        train_loss1.reset()
        train_accuracy1.reset()
        train_loss2.reset()
        train_accuracy2.reset()

        net1.train()
        net2.train()
        adjust_lr(optimizer1, lr_plan[epoch])
        adjust_lr(optimizer2, lr_plan[epoch])
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        # train this epoch
        for it, sample in enumerate(train_loader):
            s = time.time()
            # optimizer1.zero_grad()
            # optimizer2.zero_grad()
            indices = sample['index']
            x1, x2 = sample['data']
            x1, x2 = x1.to(device), x2.to(device)
            y0 = sample['label'].to(device)
            y = get_smoothed_label_distribution(y0, nc=n_classes, epsilon=cfg.epsilon)

            output1 = net1(x1)
            output2 = net2(x2)
            logits1 = output1['logits']
            logits2 = output2['logits']

            if epoch < cfg.stage1:  # warmup
                if flag[0] == 0:
                    step_flagging('stage 1')
                    flag[0] += 1
                loss1 = cross_entropy(logits1, y)
                loss2 = cross_entropy(logits2, y)
            else:  # learn label distributions
                if flag[1] == 0:
                    step_flagging('stage 2')
                    flag[1] += 1
                with torch.no_grad():
                    cce_losses1 = cross_entropy(logits1, y, reduction='none')
                    cce_losses2 = cross_entropy(logits2, y, reduction='none')
                    losses1 = cce_losses1
                    losses2 = cce_losses2
                    # ent_losses1 = entropy_loss(logits1, reduction='none')
                    # ent_losses2 = entropy_loss(logits2, reduction='none')
                    # losses1 = cce_losses1 + ent_losses1  # (N)
                    # losses2 = cce_losses2 + ent_losses2  # (N)
                    sample_selection = sample_selector(losses1, losses2, drop_rate_scheduler[epoch])

                # for selected "clean" samples, train in a co-teaching manner
                logits_clean1 = logits1[sample_selection['clean2']]
                logits_clean2 = logits2[sample_selection['clean1']]
                y_clean1 = y[sample_selection['clean2']]
                y_clean2 = y[sample_selection['clean1']]
                losses_clean1 = cross_entropy(logits_clean1, y_clean1, reduction='none') + entropy_loss(logits_clean1, reduction='none')  # (Nc1)
                losses_clean2 = cross_entropy(logits_clean2, y_clean2, reduction='none') + entropy_loss(logits_clean2, reduction='none')  # (Nc2)
                loss_c1_1 = losses_clean1.mean()
                loss_c2_1 = losses_clean2.mean()

                # for selected "unclean" samples, train in a label distribution learning manner (exchange again)
                y_t1 = labels2learn1[indices, :].clone().to(device)
                y_t2 = labels2learn2[indices, :].clone().to(device)
                y_t1.requires_grad = True
                y_t2.requires_grad = True
                y_d1 = F.softmax(y_t1, dim=1) + 1e-8
                y_d2 = F.softmax(y_t2, dim=1) + 1e-8
                logits_unclean1 = logits1[sample_selection['unclean2']]
                logits_unclean2 = logits2[sample_selection['unclean1']]
                y_d_unclean1 = y_d1[sample_selection['unclean2']]
                y_d_unclean2 = y_d2[sample_selection['unclean1']]

                w1 = np.random.beta(cfg.phi, cfg.phi, logits_unclean1.size(0))
                w2 = np.random.beta(cfg.phi, cfg.phi, logits_unclean2.size(0))
                w1 = x1.new(w1).view(logits_unclean1.size(0), 1, 1, 1)
                w2 = x2.new(w2).view(logits_unclean2.size(0), 1, 1, 1)
                idx1 = np.random.choice(sample_selection['clean2'].cpu().numpy(), logits_unclean1.size(0), replace=False if sample_selection['clean2'].size(0) >= logits_unclean1.size(0) else True)
                idx1 = torch.tensor(idx1).to(device)
                idx2 = np.random.choice(sample_selection['clean1'].cpu().numpy(), logits_unclean2.size(0), replace=False if sample_selection['clean1'].size(0) >= logits_unclean2.size(0) else True)
                idx2 = torch.tensor(idx2).to(device)
                mixed_x1 = w1 * x1[sample_selection['unclean2']] + (1-w1) * x1[idx1]
                mixed_x2 = w2 * x2[sample_selection['unclean1']] + (1-w2) * x2[idx2]
                mixed_y1 = w1 * y_d_unclean1 + (1-w1) * y_d1[idx1]
                mixed_y2 = w2 * y_d_unclean2 + (1-w2) * y_d2[idx2]

                mixed_output1 = net1(mixed_x1)
                mixed_output2 = net2(mixed_x2)
                mixed_logits1 = mixed_output1['logits']
                mixed_logits2 = mixed_output2['logits']
                loss_c1_2 = kl_div(F.softmax(mixed_logits1, dim=1) + 1e-8, mixed_y1).mean()
                loss_c2_2 = kl_div(F.softmax(mixed_logits2, dim=1) + 1e-8, mixed_y2).mean()

                loss_c1 = loss_c1_1 + loss_c1_2 * cfg.beta
                loss_c2 = loss_c2_1 + loss_c2_2 * cfg.beta

                # consistency loss
                loss_o1 = cross_entropy(F.softmax(y_t1[sample_selection['clean2']], dim=1), y[sample_selection['clean2']])
                loss_o2 = cross_entropy(F.softmax(y_t2[sample_selection['clean1']], dim=1), y[sample_selection['clean1']])

                # self-supervised loss
                target1 = output2['embeddings'].clone().detach()
                target2 = output1['embeddings'].clone().detach()
                loss_self1 = regression_loss(output1['logits'], target1).mean()
                loss_self2 = regression_loss(output2['logits'], target2).mean()

                # final loss
                loss1 = (1 - cfg.alpha) * loss_c1 + cfg.alpha * loss_o1 + loss_self1 * cfg.gamma
                loss2 = (1 - cfg.alpha) * loss_c2 + cfg.alpha * loss_o2 + loss_self2 * cfg.gamma

            train_acc1 = accuracy(logits1, y0, topk=(1,))
            train_acc2 = accuracy(logits2, y0, topk=(1,))
            train_loss1.update(loss1.item(), x1.size(0))
            train_loss2.update(loss2.item(), x2.size(0))
            train_accuracy1.update(train_acc1[0], x1.size(0))
            train_accuracy2.update(train_acc2[0], x2.size(0))

            if cfg.use_fp16:
                with amp.scale_loss(loss1, optimizer1) as scaled_loss1:
                    scaled_loss1.backward()
                with amp.scale_loss(loss2, optimizer2) as scaled_loss2:
                    scaled_loss2.backward()
            else:
                loss1.backward()
                loss2.backward()

            optimizer1.step()
            optimizer2.step()
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            if epoch >= cfg.stage1:
                y_t1.data.sub_(cfg.lmd * y_t1.grad.data)
                y_t2.data.sub_(cfg.lmd * y_t2.grad.data)
                labels2learn1[indices, :] = y_t1.detach().clone().cpu().data
                labels2learn2[indices, :] = y_t2.detach().clone().cpu().data
                del y_t1, y_t2, target1, target2

            iter_time.update(time.time() - s, 1)
            if (cfg.log_freq is not None and (it + 1) % cfg.log_freq == 0) or (it + 1 == len(train_loader)):
                total_mem = torch.cuda.get_device_properties(0).total_memory / 2**30
                mem = torch.cuda.memory_reserved() / 2**30
                console_content = f"Epoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  " \
                                  f"Iter:[{it + 1:>4d}/{len(train_loader):>4d}]  " \
                                  f"Train Accuracy 1:[{train_accuracy1.avg:6.2f}]  " \
                                  f"Train Accuracy 2:[{train_accuracy2.avg:6.2f}]  " \
                                  f"Loss 1:[{train_loss1.avg:4.4f}]  " \
                                  f"Loss 2:[{train_loss2.avg:4.4f}]  " \
                                  f"GPU-MEM:[{mem:6.3f}/{total_mem:6.3f} Gb]  " \
                                  f"{iter_time.avg:6.2f} sec/iter"
                logger.debug(console_content)

        # evaluate this epoch
        test_accuracy1 = evaluate(test_loader, net1, device)
        test_accuracy2 = evaluate(test_loader, net2, device)
        if test_accuracy1 > best_accuracy1:
            best_accuracy1 = test_accuracy1
            best_epoch1 = epoch + 1
            torch.save(net1.state_dict(), f'{result_dir}/net1_best_epoch.pth')
        if test_accuracy2 > best_accuracy2:
            best_accuracy2 = test_accuracy2
            best_epoch2 = epoch + 1
            torch.save(net2.state_dict(), f'{result_dir}/net2_best_epoch.pth')

        # logging this epoch
        runtime = time.time() - start_time
        logger.info(f'epoch: {epoch + 1:>3d} | '
                    f'train loss(1/2): ({train_loss1.avg:>6.4f}/{train_loss2.avg:>6.4f}) | '
                    f'train accuracy(1/2): ({train_accuracy1.avg:>6.3f}/{train_accuracy2.avg:>6.3f}) | '
                    f'test accuracy(1/2): ({test_accuracy1:>6.3f}/{test_accuracy2:>6.3f}) | '
                    f'epoch runtime: {runtime:6.2f} sec | '
                    f'best accuracy(1/2): ({best_accuracy1:6.3f}/{best_accuracy2:6.3f}) @ epoch: ({best_epoch1:03d}/{best_epoch2:03d})')
        plot_results_cotraining(result_file=f'{result_dir}/log.txt')

    torch.save(labels2learn1, f'{result_dir}/labels_learned.pt')

    # rename results dir -----------------------------------------------------------------------------------------------------------------------------
    best_accuracy = max(best_accuracy1, best_accuracy2)
    os.rename(result_dir, f'{result_dir}-bestAcc_{best_accuracy:.4f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--log_prefix', type=str, default='coldl')
    parser.add_argument('--log_freq', type=int, default=50)

    parser.add_argument('--net1', type=str, default='resnet18')
    parser.add_argument('--net2', type=str, default='resnet18')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--stage1', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=110)
    parser.add_argument('--lmd', type=float, default=200)
    parser.add_argument('--ldl_rate', type=float, default=0.5)
    parser.add_argument('--phi', type=float, default=0.4)
    parser.add_argument('--epsilon', type=float, default=0.5)

    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=0.1)

    args = parser.parse_args()
    config = load_from_cfg(args.config)
    override_config_items = [k for k, v in args.__dict__.items() if k != 'config' and v is not None]
    for item in override_config_items:
        config.set_item(item, args.__dict__[item])
    config.log = f'{config.net1}-{config.net2}-{config.log_prefix}'

    print(config)
    return config


if __name__ == '__main__':
    params = parse_args()
    dev = set_device(params.gpu)
    script_start_time = time.time()
    main(params, dev)
    script_runtime = time.time() - script_start_time
    print(f'Runtime of this script {str(pathlib.Path(__file__))} : {script_runtime:.1f} seconds ({script_runtime/3600:.3f} hours)')
