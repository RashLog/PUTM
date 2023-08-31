import collections
import numpy as np
import torch
import iterative_graph_functions as igf
from sklearn import metrics
import scipy as sp
from scipy.stats import t

use_gpu = torch.cuda.is_available()

# ========================================
#      loading datas

class ILPC:
    def __init__(self, model, device, log_file, args):
        self.device = device
        self.n_ways = args.n_ways
        self.number_tasks = args.batch_size
        self.model = model
        self.log_file = log_file
        self.shot = args.shot
        self.n_lsamples = args.shot * args.n_ways
        self.balanced = args.balanced == 'balanced'

    def run_task(self, ndatas, labels, params):
        ndatas, n_nfeat = self.preprocess(ndatas)
        acc = trans_ilpc(params, ndatas, labels, self.n_lsamples)
        logs = {"acc": np.array(acc).reshape(-1, 1)}
        return logs

    def preprocess(self, datas):
        beta = 0.5
        nve_idx = np.where(datas.cpu().detach().numpy() < 0)
        datas[nve_idx] *= -1
        datas[:, ] = torch.pow(datas[:, ] + 1e-6, beta)
        datas[nve_idx] *= -1

        n_nfeat = datas.size(2)
        datas = scaleEachUnitaryDatas(datas)
        datas = centerDatas(datas)

        return datas, n_nfeat



def centerDatas(datas):
    # datas[:, :n_lsamples] = datas[:, :n_lsamples, :] - datas[:, :n_lsamples].mean(1, keepdim=True)
    # datas[:, :n_lsamples] = datas[:, :n_lsamples, :] / torch.norm(datas[:, :n_lsamples, :], 2, 2)[:, :, None]
    # datas[:, n_lsamples:] = datas[:, n_lsamples:, :] - datas[:, n_lsamples:].mean(1, keepdim=True)
    # datas[:, n_lsamples:] = datas[:, n_lsamples:, :] / torch.norm(datas[:, n_lsamples:, :], 2, 2)[:, :, None]

    # centre of mass of all data support + querries
    datas[:, :] = datas[:, :, :] - datas[:, :].mean(1, keepdim=True)
    datas[:, :] = datas[:, :, :] / torch.norm(datas[:, :, :], 2, 2)[:, :, None]
    return datas

def scaleEachUnitaryDatas(datas):
    # print(datas.shape)
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * t._ppf((1 + confidence) / 2., n - 1)
    return m, h

def trans_ilpc(params, X, Y, labelled_samples):
    acc = []
    for i in (range(X.shape[0])):
        if (i + 1) % 50 == 0:
            acc_mine, acc_std = mean_confidence_interval(acc)
            out = 'Task_ID: {}, accuracy ct: {:0.2f} +- {:0.2f}, shots: {}\n'.format(i, acc_mine * 100, acc_std * 100,
                                                                                     params.shot)
            print(out)
        support_features, query_features = X[i, :labelled_samples], X[i,
                                                                    labelled_samples:]  # X_pca[:labelled_samples], X_pca[labelled_samples:] #
        support_ys, query_ys = Y[i, :labelled_samples], Y[i, labelled_samples:]
        labelled_samples = support_features.shape[0]
        if params.alpha_dirichlet is None:
            if params.unbalanced == True:
                query_features, query_ys, params.no_samples = unbalancing(params, query_features, query_ys)
            else:
                params.no_samples = np.array(np.repeat(float(query_ys.shape[0] / params.n_ways), params.n_ways))
        else:
            params.no_samples = np.zeros(params.n_ways, dtype=int)
            counter = collections.Counter(query_ys.long().tolist())
            for k, v in counter.items():
                params.no_samples[k] = v
        print(params.no_samples)
        # query_ys_pred, probs, _ = igf.update_plabels(opt, support_features, support_ys, query_features)
        # P, query_ys_pred, indices = igf.compute_optimal_transport(opt, torch.Tensor(probs))
        query_ys, query_ys_pred = igf.iter_balanced_trans(params, support_features, support_ys, query_features, query_ys,
                                                          labelled_samples)
        # clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
        # clf = LogisticRegression(C=10, multi_class='auto', solver='lbfgs', max_iter=1000)
        # clf.fit(support_features, support_ys)
        # query_ys_pred = clf.predict(query_features)
        acc_task = metrics.accuracy_score(query_ys, query_ys_pred)
        acc.append(acc_task)

    return acc


# def trans_ct(opt, X, Y, labelled_samples):
#     acc = []
#     use_cuda = True
#     for i in tqdm(range(X.shape[0] + 1)):
#         if (i + 1) % 1 == 0:
#             acc_mine, acc_std = mean_confidence_interval(acc)
#             print('Task_ID: {}, accuracy ct: {:0.2f} +- {:0.2f}, shots: {}'.format(i, acc_mine * 100, acc_std * 100,
#                                                                                    n_shot))
#         support_features, query_features = X[i, :labelled_samples], X[i, labelled_samples:]
#         support_ys, query_ys = Y[i, :labelled_samples], Y[i, labelled_samples:]
#         labelled_samples = support_features.shape[0]
#         if params.unbalanced == True:
#             query_features, query_ys, opt.no_samples = unbalancing(opt, query_features, query_ys)
#         else:
#             opt.no_samples = np.array(np.repeat(float(query_ys.shape[0] / opt.n_ways), opt.n_ways))
#         if use_cuda:
#             support_features = support_features.cuda()
#             query_features = query_features.cuda()
#             support_ys = support_ys.cuda()
#             query_ys = query_ys.cuda()
#         feature_dim = support_features.size(-1)
#         prototype = support_features.view(-1, opt.n_ways, feature_dim).mean(dim=0)
#         CTNet = CT(prototype, query_features, feat_dim=feature_dim,
#                    num_iter=20,
#                    use_cuda=use_cuda,
#                    rho=0.5,
#                    alpha=0.1,
#                    lr=0.001)
#         prototype, _ = CTNet.updataPrototype(50)
#         probas = CTNet.getProbas()
#         query_ys_pred = probas.squeeze().max(dim=1)[1]
#         acc_task = torch.sum((query_ys == query_ys_pred).long()) / query_ys.shape[0]
#         acc.append(acc_task.item())
#     return mean_confidence_interval(acc)


def unbalancing(opt, query_features, query_ys):
    max = opt.n_queries
    min = opt.n_queries - opt.un_range
    no_samples = np.array(np.random.randint(min, max, size=opt.n_ways))
    no_classes = query_ys.max() + 1
    q_y = []
    q_f = []
    for i in range(no_classes):
        idx = np.where(query_ys == i)
        tmp_y, tmp_x = query_ys[idx], query_features[idx]
        # print(tmp_y[0:no_samples[i]].shape)
        q_y.append(tmp_y[:no_samples[i]])
        q_f.append(tmp_x[:no_samples[i]])
    q_y = torch.cat(q_y, dim=0)
    q_f = torch.cat(q_f, dim=0)
    # print(q_y.shape)
    return q_f, q_y, no_samples
    # print(model.weight.shape, imprinted.shape)



#
# if __name__ == '__main__':
#     # ---- data loading
#     params = parse_option()
#     n_shot = params.n_shots
#     n_ways = params.n_ways
#     n_unlabelled = params.n_unlabelled
#     n_queries = params.n_queries
#     if params.unbalanced == True:
#         params.un_range = 10
#         params.n_queries = n_queries + params.un_range
#         n_queries = params.n_queries
#     if params.alpha_dirichlet is not None:
#         n_queries = n_queries - params.un_range
#     print("n_queries:{}, unbalanced:{}, denoising_iterations:{}".format(params.n_queries,
#                                                                         params.unbalanced,
#                                                                         params.denoising_iterations))
#     n_runs = 100
#     FSLTask_im._maxRuns = n_runs
#     n_lsamples = n_ways * n_shot
#     n_usamples = n_ways * n_queries
#     n_samples = n_lsamples + n_usamples
#     dataset = params.dataset
#     distribution = 'dirichlet' if params.alpha_dirichlet is not None else 'uniform'
#
#     cfg = {
#         'shot': n_shot,
#         'ways': n_ways,
#         'queries': n_queries,
#         'tasks': n_runs,
#         'sample': distribution
#     }
#     FSLTask_im.loadDataSet(params.backbone, dataset)
#     FSLTask_im.setRandomStates(cfg)
#     ndatas, labels, query_samples = FSLTask_im.GenerateRunSet(cfg=cfg)
#     if cfg['sample'] == 'uniform':
#         ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
#         labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
#                                                                                                             n_samples)
#     elif cfg['sample'] == 'dirichlet':
#         pass
#
#     print("size of the datas...", ndatas.size())
#
#     """
#     if params.alpha_dirichlet is not None:
#         print('alpha_dirichlet =', params.alpha_dirichlet)
#         FSLTask_im._alpha = params.alpha_dirichlet
#         cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries, 'tasks': n_runs, 'sample': 'dirichlet'}
#         FSLTask_im.loadDataSet(params.backbone, dataset)
#         FSLTask_im.setRandomStates(cfg)
#         ndatas, labels, _ = FSLTask_im.GenerateRunSet(cfg=cfg)
#         labels = labels.long()
#     else:
#
#         cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
#         FSLTask.loadDataSet(params.backbone, dataset)
#         FSLTask.setRandomStates(cfg)
#         ndatas = FSLTask.GenerateRunSet(cfg=cfg)
#         # print(ndatas.shape)
#         ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
#         labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
#                                                                                                             n_samples)
#         # print(params.unbalanced)
#     """
#
#     orig_ndatas = ndatas.clone()
#     orig_labels = labels.clone()
#
#     # Power transform
#     beta = 0.5
#     # ------------------------------------PT-MAP-----------------------------------------------
#     nve_idx = np.where(ndatas.cpu().detach().numpy() < 0)
#     ndatas[nve_idx] *= -1
#     ndatas[:, ] = torch.pow(ndatas[:, ] + 1e-6, beta)
#     ndatas[nve_idx] *= -1  # return the sign
#     # ------------------------------------------------------------------------------------------
#     # print(ndatas.type())
#     n_nfeat = ndatas.size(2)
#
#     ndatas = scaleEachUnitaryDatas(ndatas)
#     ndatas = centerDatas(ndatas)
#     # ndatas = ndatas.cpu()
#     # labels = labels.cpu()
#
#     # print("size of the datas...", ndatas.size())
#
#     if params.algorithm == 'ilpc':
#         balanced = 'uniform' if params.alpha_dirichlet is None else 'dirichlet%d' % params.alpha_dirichlet
#         save_dir = os.path.join('results', params.backbone, params.dataset, balanced, str(params.n_shots))
#         os.makedirs(save_dir, exist_ok=True)
#         with open(os.path.join(save_dir, 'ilpc_results.txt'), 'a+') as f:
#             f.write('Class distribution: {}\n'.format(labels[0]))
#         acc_mine, acc_std = trans_ilpc(params, ndatas, labels, n_lsamples, save_dir)
#
#         output = 'DATASET: {}, final accuracy ilpc: {:0.2f} +- {:0.2f}, shots: {}, queries: {}\n'.format(dataset,
#                                                                                                          acc_mine * 100,
#                                                                                                          acc_std * 100,
#                                                                                                          n_shot,
#                                                                                                          n_queries)
#
#
#         print(output)
#     else:
#         print('Algorithm not supported!')
