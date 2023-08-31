import os
import torch
import torch.nn as nn
import numpy as np
import wrn_mixup_model
import argparse
from utils import warp_tqdm, save_pickle, load_pickle
from models.WideResNet import wideres
from models.ResNet import resnet18
from datasets import get_dataset
from torch.utils.data import DataLoader


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x, feature):
        return self.module(x, feature)


def extract_mean_features(args, model=None, train_loader=None, device=None):
    """
        inputs:
            model : The loaded model containing the feature extractor
            train_loader : Train data loader
            args : arguments
            logger : logger object
            device : GPU device

        returns:
            out_mean : Training data features mean
    """

    # Load features from memory if previously saved ...
    save_dir = '.'
    train_mean_file = '_'.join([args.arch, args.dataset, 'train_mean.plk'])
    train_mean_file = os.path.join('train_mean', train_mean_file)
    print('train_mean file: ', train_mean_file)
    filepath_mean = os.path.join(save_dir, train_mean_file)

    # get training mean
    if not os.path.isfile(filepath_mean):
        os.makedirs(save_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            out_mean, fc_out_mean = [], []
            for i, batch in enumerate(warp_tqdm(train_loader, False)):
                inputs, _, _ = batch
                inputs = inputs.to(device)
                outputs, fc_outputs = model(inputs, feature=True)
                out_mean.append(outputs.cpu().data.numpy())
                if fc_outputs is not None:
                    fc_out_mean.append(fc_outputs.cpu().data.numpy())
            out_mean = np.concatenate(out_mean, axis=0).mean(0)
            if len(fc_out_mean) > 0:
                fc_out_mean = np.concatenate(fc_out_mean, axis=0).mean(0)
            else:
                fc_out_mean = -1
            save_pickle(os.path.join(save_dir, train_mean_file), [out_mean, fc_out_mean])
            return  torch.from_numpy(out_mean),  torch.from_numpy(fc_out_mean)
    else:
        print('Train mean file already exists.')
        out_mean, fc_out_mean = load_pickle(os.path.join(save_dir, train_mean_file))
        return torch.from_numpy(out_mean), torch.from_numpy(fc_out_mean)

def parse_args():
  parser = argparse.ArgumentParser(description='Extract trainset mean.')
  parser.add_argument('--modelfile', type=str, required=True)
  parser.add_argument('--arch', type=str, required=True)
  parser.add_argument('--dataset', type=str, required=True)
  args = parser.parse_args()
  return args

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()
    modelfile = args.modelfile
    if args.arch == 'resnet18':
      model = resnet18(num_classes=64)
    if args.arch =='wideres':
      model = wideres(num_classes=351)
    model.to(device)

    checkpoint = torch.load(modelfile, map_location=device)
    state = checkpoint['state_dict']
    state_keys = list(state.keys())
    if 'module' in state_keys[0]:
        model = WrappedModel(model)
    model_dict_load = model.state_dict()
    model_dict_load.update(state)
    model.load_state_dict(model_dict_load)
    model.eval()

    if args.dataset == 'mini':
      args.dataset_path = '/content/Alpha-Tim_Data/mini_imagenet'
    elif args.dataset == 'cub':
      args.dataset_path = '/content/Alpha-Tim_Data/CUB/CUB_200_2011'
    else:
      raise NotImplementedError('No such dataset: ', args.dataset)

    args.split_dir = os.path.join('./split', args.dataset)
    args.enlarge = True

    trainset = get_dataset('train', args, aug=False, out_name=False)
    base_loader = DataLoader(trainset, batch_size=128, shuffle=False, pin_memory=True)
    # loadfile = args.split
    # datamgr = SimpleDataManager(84, batch_size=128)
    # base_loader = datamgr.get_data_loader(loadfile, aug=False)

    extract_mean_features(args, model, base_loader, device)
