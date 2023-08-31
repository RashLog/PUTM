import os
import pickle
import numpy as np
import torch
import math

features_dict = {}

# WRN-28-10 S2M2
features_dict['wrn_s2m2'] = {"mini": "../features/wrn_s2m2/mini/novel.plk",
                        "tiered": "../features/wrn_s2m2/tiered/novel.plk",
                        "cub": "../features/wrn_s2m2/cub/novel.plk",
                        "cifar": "../features/wrn_s2m2/cifar/novel.plk",
                        "cross": None}
features_dict['wideres'] = {"mini": "../features/wideres/mini/novel.plk",
                            "tiered":"../features/wideres/tiered/output.plk",
                            }
features_dict['resnet18'] = {"mini": "../features/resnet18/mini/novel.plk",
                             "cub": "../features/resnet18/cub/novel.plk",
                             }
_cacheDir = "./cache"
_maxRuns = 1000
_min_examples = -1
_alpha = 2
# ========================================================
#   Module internal functions and variables

_randStates = None
_rsCfg = None


def get_dirichlet_query_dist(alpha, n_tasks, n_ways, q_shots):
    alpha = np.full(n_ways, alpha)
    prob_dist = np.random.dirichlet(alpha, n_tasks)
    return convert_prob_to_samples(prob=prob_dist, q_shot=q_shots)


def convert_prob_to_samples(prob, q_shot):
    prob = prob * q_shot
    for i in range(len(prob)):
        if sum(np.round(prob[i])) > q_shot:
            while sum(np.round(prob[i])) != q_shot:
                idx = 0
                for j in range(len(prob[i])):
                    frac, whole = math.modf(prob[i, j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[i, idx] = np.floor(prob[i, idx])
            prob[i] = np.round(prob[i])
        elif sum(np.round(prob[i])) < q_shot:
            while sum(np.round(prob[i])) != q_shot:
                idx = 0
                for j in range(len(prob[i])):
                    frac, whole = math.modf(prob[i, j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[i, idx] = np.ceil(prob[i, idx])
            prob[i] = np.round(prob[i])
        else:
            prob[i] = np.round(prob[i])
    return prob.astype(int)


def _load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        labels = [np.full(shape=len(data[key]), fill_value=key)
                  for key in data]
        data = [features for key in data for features in data[key]]
        dataset = dict()
        dataset['data'] = torch.FloatTensor(np.stack(data, axis=0))
        dataset['labels'] = torch.LongTensor(np.concatenate(labels))
        return dataset


# =========================================================
#    Callable variables and functions from outside the module

data = None
labels = None
dsName = None


def loadDataSet(backbone, dsname):
    if backbone not in features_dict.keys():
        raise NameError('Unknown backbone: {}'.format(backbone))
    _datasetFeaturesFiles = features_dict[backbone]
    if dsname not in _datasetFeaturesFiles:
        raise NameError('Unknwown dataset: {}'.format(dsname))

    global dsName, data, labels, _randStates, _rsCfg, _min_examples
    dsName = dsname
    _randStates = None
    _rsCfg = None

    # Loading data from files on computer
    # home = expanduser("~")
    print("Loading features from %s ..." % _datasetFeaturesFiles[dsname])
    dataset = _load_pickle(_datasetFeaturesFiles[dsname])
    dataset['data'] = dataset['data'].squeeze()

    # Computing the number of items per class in the dataset
    _min_examples = dataset["labels"].shape[0]
    for i in range(dataset["labels"].shape[0]):
        if torch.where(dataset["labels"] == dataset["labels"][i])[0].shape[0] > 0:
            _min_examples = min(_min_examples, torch.where(
                dataset["labels"] == dataset["labels"][i])[0].shape[0])
    print("Guaranteed number of items per class: {:d}\n".format(_min_examples))

    # Generating data tensors
    data = torch.zeros((0, _min_examples, dataset["data"].shape[1]))
    labels = dataset["labels"].clone()
    while labels.shape[0] > 0:
        indices = torch.where(dataset["labels"] == labels[0])[0]
        data = torch.cat([data, dataset["data"][indices, :]
        [:_min_examples].view(1, _min_examples, -1)], dim=0)
        indices = torch.where(labels != labels[0])[0]
        labels = labels[indices]
    print("Total of {:d} classes, {:d} elements each, with dimension {:d}\n".format(
        data.shape[0], data.shape[1], data.shape[2]))


def GenerateRun(iRun, cfg, regenRState=False, generate=True):
    global _randStates, data, _min_examples
    if not regenRState:
        np.random.set_state(_randStates[iRun])

    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    shuffle_indices = np.arange(_min_examples)
    dataset = None
    dataset_s = None
    dataset_q = None
    label_s = None
    label_q = None
    if generate:
        if cfg['sample'] == 'uniform':
            dataset = torch.zeros(
                (cfg['ways'], cfg['shot'] + cfg['queries'], data.shape[2]))
        elif cfg['sample'] == 'dirichlet':
            dataset_s = torch.zeros(cfg['ways'], cfg['shot'], data.shape[2])
            dataset_q = torch.zeros(cfg['ways'] * cfg['queries'], data.shape[2])
            label_s = torch.zeros(cfg['ways'], cfg['shot'])
            label_q = torch.zeros(cfg['ways'] * cfg['queries'])
        else:
            raise NotImplementedError
    query_samples = get_dirichlet_query_dist(_alpha, 1, cfg['ways'], cfg['ways'] * cfg['queries'])[0]
    if len(query_samples) < 5:
        print('debugging...')
    query_samples_ = get_dirichlet_query_dist(_alpha, 1, cfg['ways'], cfg['ways'] * cfg['queries'])[0]
    while max(query_samples) > _min_examples - cfg['shot']:
        query_samples = get_dirichlet_query_dist(_alpha, 1, cfg['ways'], cfg['ways'] * cfg['queries'])[0]
    begin_idx = 0
    for i in range(cfg['ways']):
        shuffle_indices = np.random.permutation(shuffle_indices)
        if generate:
            if cfg['sample'] == 'uniform':
                dataset[i] = data[classes[i], shuffle_indices,
                             :][:cfg['shot'] + cfg['queries']]
            elif cfg['sample'] == 'dirichlet':
                end_idx = begin_idx + query_samples[i]
                dataset_s[i][:cfg['shot']] = data[classes[i], shuffle_indices, :][:cfg['shot']]
                label_s[i][: cfg['shot']] = i * torch.ones(1, cfg['shot'])
                dataset_q[begin_idx: end_idx] = data[classes[i], shuffle_indices, :][
                                                cfg['shot']: cfg['shot'] + query_samples[i]]
                label_q[begin_idx: end_idx] = i * torch.ones(1, query_samples[i])
                begin_idx = end_idx
            else:
                raise NotImplementedError
    if cfg['sample'] == 'uniform':
        return dataset
    elif cfg['sample'] == 'dirichlet':
        dataset = torch.cat((dataset_s.view(cfg['ways'] * cfg['shot'], -1), dataset_q), dim=0)
        label = torch.cat((label_s.view(cfg['ways'] * cfg['shot'], -1), label_q.unsqueeze(1)), dim=0).squeeze()
        return dataset, label, torch.from_numpy(query_samples_ + 1)


def ClassesInRun(iRun, cfg):
    global _randStates, data
    np.random.set_state(_randStates[iRun])

    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    return classes


def setRandomStates(cfg):
    global _randStates, _maxRuns, _rsCfg
    if _rsCfg == cfg:
        return

    rsFile = os.path.join(_cacheDir, "RandStates_{}_s{}_q{}_w{}_t{}".format(
        dsName, cfg['shot'], cfg['queries'], cfg['ways'], cfg['tasks']))
    if not os.path.exists(rsFile):
        print("{} does not exist, regenerating it...".format(rsFile))
        np.random.seed(0)
        _randStates = []
        for iRun in range(_maxRuns):
            _randStates.append(np.random.get_state())
            GenerateRun(iRun, cfg, regenRState=False, generate=True)
        torch.save(_randStates, rsFile)
    else:
        print("reloading random states from file....")
        _randStates = torch.load(rsFile)
    _rsCfg = cfg


def GenerateRunSet(start=None, end=None, cfg=None):
    global dataset, _maxRuns
    if start is None:
        start = 0
    if end is None:
        end = _maxRuns
    if cfg is None:
        cfg = {"shot": 1, "ways": 5, "queries": 15}

    setRandomStates(cfg)
    print("generating task from {} to {}".format(start, end))

    dataset = torch.zeros(
        (end - start, cfg['ways'], cfg['shot'] + cfg['queries'], data.shape[2]))
    label = None
    query_samples = None
    if cfg['sample'] == 'dirichlet':
        dataset = torch.zeros(
            (end - start, cfg['ways'] * (cfg['shot'] + cfg['queries']), data.shape[2]))
        label = torch.zeros(
            (end - start, cfg['ways'] * (cfg['shot'] + cfg['queries'])))
        query_samples = torch.zeros(
            (end - start, cfg['ways']))
    for iRun in range(end - start):
        if cfg['sample'] == 'uniform':
            dataset[iRun] = GenerateRun(start + iRun, cfg)
        elif cfg['sample'] == 'dirichlet':
            dataset[iRun], label[iRun], query_samples[iRun] = GenerateRun(start + iRun, cfg)
    return dataset, label, query_samples


# define a main code to test this module
if __name__ == "__main__":
    print("Testing Task loader for Few Shot Learning")
    loadDataSet('miniimagenet')

    cfg = {"shot": 1, "ways": 5, "queries": 15}
    setRandomStates(cfg)

    run10 = GenerateRun(10, cfg)
    print("First call:", run10[:2, :2, :2])

    run10 = GenerateRun(10, cfg)
    print("Second call:", run10[:2, :2, :2])

    ds = GenerateRunSet(start=2, end=12, cfg=cfg)
    print("Third call:", ds[8, :2, :2, :2])
    print(ds.size())
