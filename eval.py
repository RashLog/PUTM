import sys
sys.path.append('..')
import random
import os
import numpy as np
import torch
import torch.nn as nn
import FSLTask_im
import argparse
import wrn_mixup_model
from methods.baseline import Baseline, Baseline_PlusPlus
from methods.tim import ALPHA_TIM, TIM_GD
from methods.bdcspn import BDCSPN
from methods.entropy_min import Entropy_min
from methods.laplacianshot import LaplacianShot
from methods.protonet import ProtoNet
from methods.pt_map import PT_MAP
from methods.simpleshot import SimpleShot
from methods.ilpc import ILPC
from methods.bavardage import BAVARDAGE
from methods.putm import PUTM
from utils import load_cfg_from_cfg_file, merge_cfg_from_list
from models.WideResNet import wideres
from models.ResNet import resnet18

distribution = None

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument('--base_config', type=str, required=True, help='Base config file')
    parser.add_argument('--method_config',  type=str, required=True, help='Method config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    method_config = args.method_config
    assert args.base_config is not None
    cfg = load_cfg_from_cfg_file(args.base_config)
    cfg.update(load_cfg_from_cfg_file(args.method_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    for arg in args.__dict__.keys():
        if arg not in ['base_config', 'method_config', 'opts']:
            cfg.update({arg: args.__dict__[arg]})
    assert method_config.split("/")[1] == cfg.balanced
    return cfg

def centerDatas(datas):
    # datas[:, :n_lsamples] = datas[:, :n_lsamples, :] - datas[:, :n_lsamples].mean(1, keepdim=True)
    # datas[:, :n_lsamples] = datas[:, :n_lsamples, :] / torch.norm(datas[:, :n_lsamples, :], 2, 2)[:, :, None]

    # datas[:, n_lsamples:] = datas[:, n_lsamples:, :] - datas[:, n_lsamples:].mean(1, keepdim=True)
    # datas[:, n_lsamples:] = datas[:, n_lsamples:, :] / torch.norm(datas[:, n_lsamples:, :], 2, 2)[:, :, None]

    datas[:] -= datas.mean(1, keepdim=True)

    return datas


def scaleEachUnitaryDatas(datas):
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms


def QRreduction(datas):
    ndatas = torch.linalg.qr(datas.permute(0, 2, 1), 'reduced').R
    ndatas = ndatas.permute(0, 2, 1)
    return ndatas


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_method_builder(model, device, log_file, args):
    # Initialize method classifier builder
    method_info = {'model': model, 'device': device, 'log_file': log_file, 'args': args}
    if args.method == 'ALPHA-TIM':
        method_builder = ALPHA_TIM(**method_info)
    elif args.method == 'TIM-GD':
        method_builder = TIM_GD(**method_info)
    elif args.method == 'LaplacianShot':
        method_builder = LaplacianShot(**method_info)
    elif args.method == 'BDCSPN':
        method_builder = BDCSPN(**method_info)
    elif args.method == 'SimpleShot':
        method_builder = SimpleShot(**method_info)
    elif args.method == 'Baseline':
        method_builder = Baseline(**method_info)
    elif args.method == 'Baseline++':
        method_builder = Baseline_PlusPlus(**method_info)
    elif args.method == 'PT-MAP':
        method_builder = PT_MAP(**method_info)
    elif args.method == 'ProtoNet':
        method_builder = ProtoNet(**method_info)
    elif args.method == 'Entropy-min':
        method_builder = Entropy_min(**method_info)
    elif args.method == 'iLPC':
        method_builder = ILPC(**method_info)
    elif args.method == 'BAVARDAGE':
        method_builder = BAVARDAGE(**method_info)
    elif args.method == 'PUTM':
        method_builder = PUTM(**method_info)
    else:
        raise ValueError("Method must be in ['TIM_GD', 'ALPHA_TIM', 'LaplacianShot', 'BDCSPN', 'SimpleShot', 'Baseline', 'Baseline++', 'PT-MAP', 'Entropy_min']")
    return method_builder

def compute_confidence_interval(data, axis=0):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a, axis=axis)
    std = np.std(a, axis=axis)
    pm = 1.96 * (std / np.sqrt(a.shape[axis]))
    return m, pm

def get_tasks(n_ways, n_shot, n_queries, n_runs, backbone, dataset, distribution):
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    # distribution = 'dirichlet'  # uniform or dirichlet

    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries, 'tasks': n_runs, 'sample': distribution}
    FSLTask_im.loadDataSet(backbone, dataset)
    FSLTask_im.setRandomStates(cfg)
    ndatas, labels, query_samples = FSLTask_im.GenerateRunSet(cfg=cfg)
    if cfg['sample'] == 'uniform':
        ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
        labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,                                                                                                    n_samples)
    elif cfg['sample'] == 'dirichlet':
        pass

    print("size of the datas...", ndatas.size())

    return ndatas, labels

def get_model(args, backbone, modelfile):
    if backbone == 'wideres':
      net = wideres(num_classes=args.num_classes)
    elif backbone == 'resnet18':
      net = resnet18(num_classes=args.num_classes)
    elif backbone == 'WRN':
      net = wrn_mixup_model.wrn28_10(num_classes=200, loss_type='dist')
    else:
      pass
    checkpoint = torch.load(modelfile, map_location=device)
    state = checkpoint['state_dict']
    state_keys = list(state.keys())
    if 'module' in state_keys[0]:
        net = WrappedModel(net)
    model_dict_load = net.state_dict()
    model_dict_load.update(state)
    net.load_state_dict(model_dict_load)
    return net

if __name__ == '__main__':
    fix_seed(2023)

    args = parse_args()
    if args.method not in ["iLPC", "BAVARDAGE"]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    n_ways = args.n_ways
    n_queries = args.n_query
    n_runs = args.number_tasks
    FSLTask_im._maxRuns = n_runs
    FSLTask_im._alpha = args.alpha_dirichlet
    backbone = args.arch
    dataset = args.dataset
    distribution = 'dirichlet' if args.balanced == 'dirichlet' else 'uniform'
    print("backbone:{}, dataset:{}, distribution:{}, dirichlet:{}".format(backbone, dataset, distribution, args.alpha_dirichlet))

    for n_shot in args.shots:
        args.__dict__.update({'shot': n_shot})
        ndatas, labels = get_tasks(n_ways, n_shot, n_queries, n_runs, backbone, dataset, distribution)
        ndatas = ndatas.to(device)
        labels = labels.to(device)

        # model = get_model(args, backbone, modelfile)
        # model.eval()
        model = None

        method = get_method_builder(model, device, None, args)

        logs = method.run_task(ndatas, labels, args)

        acc_mean, acc_conf = compute_confidence_interval(logs['acc'][:, -1])
        
        with open("results.txt", "a+") as f:
            f.write(f'{dataset}, {backbone}, {args.method}, {n_shot}, {distribution}, {args.alpha_dirichlet}, {acc_mean:.3f}\n')

