import random
import numpy as np
import torch
import torch.optim as optim
import math
import torch.nn.functional as F
import time
import argparse
import os
import pickle
from tqdm import tqdm
from torch import nn

class BNavigator(nn.Module):
    def __init__(self, batch_size, hidden=512, dim=256):
        super(BNavigator, self).__init__()
        hidden_dims = [dim, hidden, hidden // 2, 1]
        self.params = []
        self.bns = []
        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1]
            w = nn.Parameter(torch.randn((batch_size, in_dim, out_dim)))
            b = nn.Parameter(torch.zeros((batch_size, 1, out_dim)))
            nn.init.xavier_normal_(w.data)
            self.register_parameter(f"w{i+1}", w)
            self.register_parameter(f"b{i+1}", b)
            self.params.append(w)
            self.params.append(b)
            # if i < len(hidden_dims) - 2:
            # self.bns.append(nn.BatchNorm1d(out_dim))

    def reset(self):
        for i, p in enumerate(self.params):
            if p.shape[1] > 1:
                self.params[i] = torch.randn_like(p).requires_grad_(True)
                nn.init.xavier_normal_(p.data)
            else:
                self.params[i] = torch.zeros_like(p).requires_grad_(True)

    def forward(self, x):
        n_layer = len(self.params) // 2
        for i in range(n_layer):
            x = torch.bmm(x, self.params[2 * i]) + self.params[2 * i + 1]
            if i < n_layer - 1:
                x = F.leaky_relu(x)
        return x

    def to_device(self, device):
        for i, param in enumerate(self.params):
            self.params[i] = self.params[i].to(device)
            self.params[i].requires_grad = True

class PUTM:
    def __init__(self, model, device, log_file, args):
        self.device = device
        self.n_ways = args.n_ways
        self.n_shot = args.shot
        self.n_runs = args.batch_size
        self.alpha = args.alpha[0] if args.shot == 1 else args.alpha[1]
        self.beta = args.beta
        self.lam = args.lam
        self.n_queries = args.n_query
        self.n_sum_query = args.n_query * args.n_ways
        self.n_epochs = args.n_epochs
        self.model = model
        self.log_file = log_file
        self.n_lsample = args.n_ways * args.shot
        self.tol = self.n_sum_query + self.n_lsample

    def run_task(self, ndatas, labels, args):
        bnav = BNavigator(batch_size=self.n_runs, hidden=args.hidden, dim=self.n_ways-1 if args.use_plda else self.tol)
        bnav.to(self.device)
        optim_bnav = torch.optim.Adam(bnav.parameters(), lr=args.lr, betas=(0.5, 0.99))
        
        """
        ds = getRunSet(n_shots=args.shot,
                       n_ways=args.n_ways,
                       n_queries=args.n_query,
                       preprocess=args.preprocess,
                       dataset=args.dataset,
                       model=args.arch,
                       data=ndatas.clone(),
                       labels=labels)
        ds.printState()
        """
        

        #ndatas = ndatas - ds.base_data.mean(dim=0, keepdim=True)
        ndatas = preprocess(ndatas, self.beta)
        #ndatas = ds.data
        if args.use_plda:
            ndatas = ds.rdata_b.cuda()
        

        #scalingValues = ds.e_Swb_Udata.cuda().pow(-0.5).clamp(max=args.clp)
        scalingValues = None

        model = GaussianModel(self.n_ways, self.lam, bnav, optim_bnav, ndatas, labels, self.n_lsample, args)
        
        
        if args.use_km:
            model.initFromLabelledDatas()
            mus, probs  = soft_km(ndatas, labels, args.T_km, model.mus, self.n_lsample)
            model.mus = mus if not args.use_plda else None
        else:
            probs = None

        optim = MAP(bnav, labels, self.n_lsample, self.n_runs, self.alpha, args.lr, probs, scalingValues, args.use_plda)

        optim.verbose = True
        optim.progressBar = False

        acc_test, m, pm = optim.loop(model, self.n_epochs)


        logs = {"acc": acc_test.view(-1, 1).cpu().numpy()}
        return logs



def init_wights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def soft_km(X, Y, T, mus, n_lsamples, nIter=20):
    for i in range(nIter):
        dist2 = (X.unsqueeze(2) - mus.unsqueeze(1)).norm(dim=3).pow(2)
        probas = F.softmax(-dist2*T, dim=2)
        train_labels = Y[:,:n_lsamples].long()
        probas[:,:n_lsamples] = 0
        probas[:,:n_lsamples].scatter_(2, train_labels.unsqueeze(2), 1)
        mus = probas.permute(0,2,1).matmul(X).div(probas.sum(dim=1).unsqueeze(2))

    return mus, probas


def centerDatas(datas, n_lsamples):
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] - datas[:, :n_lsamples].mean(1, keepdim=True)
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] / torch.norm(datas[:, :n_lsamples, :], 2, 2)[:, :, None]
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] - datas[:, n_lsamples:].mean(1, keepdim=True)
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] / torch.norm(datas[:, n_lsamples:, :], 2, 2)[:, :, None]
    #    datas[:] -= datas.mean(1, keepdim=True)
    return datas


def scaleEachUnitaryDatas(datas):
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms

def QRreduction(datas):
    ndatas = torch.linalg.qr(datas.permute(0, 2, 1), 'reduced').R
    ndatas = ndatas.permute(0, 2, 1)
    return ndatas

def preprocess(data, beta):
    data[:, ] = torch.pow(data[:, ] + 1e-6, beta)
    data = QRreduction(data)
    # n_nfeat = data.size(2)
    data = scaleEachUnitaryDatas(data)
    # print(torch.abs(data -  F.normalize(data,dim=2)).sum())
    #    ndatas = centerDatas(ndatas)
    # data = F.normalize(data, dim=2)

    return data

class Model:
    def __init__(self, n_ways):
        self.n_ways = n_ways


# ---------  GaussianModel
class GaussianModel(Model):
    def __init__(self, n_ways, lam, bnav, optim_bnav, ndatas, labels, n_lsamples, args):
        super(GaussianModel, self).__init__(n_ways)
        self.mus = None  # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam
        self.c = None
        self.flag = False
        self.bnav = bnav
        self.optim_bnav = optim_bnav
        self.ndatas = ndatas
        self.orig_ndatas = ndatas.clone()
        self.labels = labels
        self.args = args
        self.n_lsamples = n_lsamples
        self.args.n_runs = args.batch_size

    def cuda(self):
        self.mus = self.mus.cuda()

    def initFromLabelledDatas(self, probas=None):
        args = self.args
        if self.args.balanced == "dirichlet":
            self.mus = self.ndatas[:, :self.n_lsamples].reshape(args.n_runs, args.n_ways, args.shot, -1).mean(2)
        elif self.args.balanced == "balanced":
            self.mus = self.ndatas[:, :self.n_lsamples].reshape(args.n_runs, args.shot, args.n_ways, -1).mean(1)
        
        if probas is not None:
            print("probas.shape", probas.shape)
            self.mus = probas.permute(0, 2, 1).matmul(self.ndatas[:, :]).div(probas.sum(dim=1).unsqueeze(2))
        
        print(f"[Gaussian Model.initFromLabel...]: mus.shape={self.mus.shape}")

    def initDistribution(self):
        args = self.args
        self.c = (torch.ones(args.n_runs, args.n_ways) / (args.shot))  # [100,1]

    def updateDistribution(self, probas):
        self.c = probas
        self.flag = True

    def estimateFromMask(self, mask):
        train_labels = self.labels[:, :self.n_lsamples].long()
        gt = torch.zeros(mask.size(0), self.n_lsamples, mask.size(-1)).cuda().scatter_(2, train_labels.unsqueeze(2), 1)
        mask = torch.cat([gt, mask], dim=1)
        emus = mask.permute(0, 2, 1).matmul(self.ndatas[:, :]).div(mask.sum(dim=1).unsqueeze(2))
        return emus

    def updateFromEstimate(self, estimate, alpha):
        with torch.no_grad():
            Dmus = estimate - self.mus
            self.mus = self.mus + alpha * (Dmus)

    def ct_loss(self, f_x, f_y):
        args = self.args
        # last_m = None
        self.bnav.train()
        m_forward = None
        m_backward = None
        for epoch in range(1, args.epochs + 1):
            self.optim_bnav.zero_grad()
            mse_n = (f_x[:, None] - f_y).pow(2)
            
            cost = mse_n.sum(-1)
            d = self.bnav(mse_n.view(args.n_runs, 75 * 5, -1)).squeeze().mul(-1)
            d = d.view(args.n_runs, 75, 5)

            m_forward = torch.softmax(d, dim=2)

            m_backward = torch.softmax(d, dim=1)

            gen_loss = args.rho * (cost * m_forward).sum(2).mean(1) + (1 - args.rho) * (cost * m_backward).sum(1).mean(1)
            # gen_loss.backward(torch.ones_like(gen_loss), retain_graph=False)

            balance_navigator = args.rho * m_forward + (1 - args.rho) * m_backward
            d_balance = torch.sum(balance_navigator, dim=1)
            d_target = torch.ones(args.n_runs, 5) * 0.2
            d_balance = d_balance.to(mse_n.device)
            d_target = d_target.to(mse_n.device)
            mse_balance = (d_balance - d_target).pow(2)
            cost_balance = mse_balance.mean(1)

            total_loss = args.theta * cost_balance + (1 - args.theta) * gen_loss
            total_loss.backward(torch.ones_like(total_loss), retain_graph=False)

            self.optim_bnav.step()

            # if last_m is not None:
            #     with torch.no_grad():
            #         diff = torch.abs(m_forward - last_m).sum()
            #         if epoch % 10 == 0:
            #             print("epoch: %d, diff: %.2f" % (epoch, diff))
            # last_m = m_forward
        return args.rho * m_forward + (1 - args.rho) * m_backward, m_forward.detach(), m_backward.detach()

    def getProbas(self):
        ndatas_ls = self.ndatas[:, self.n_lsamples:, :].unsqueeze(2)
        m_mixed, m_for, m_back = self.ct_loss(self.mus, ndatas_ls)
        return m_for


# =========================================
#    MAP
# =========================================

class MAP:
    def __init__(self, bnav, labels, n_lsamples, n_runs, alpha=None, lr=None, probs=None, scalingValues=None, use_plda=False):

        self.verbose = False
        self.progressBar = False
        self.alpha = alpha
        self.lr = lr
        self.bnav = bnav
        self.labels = labels
        self.n_lsamples = n_lsamples
        self.n_runs = n_runs
        self.probas = probs
        self.adr = Basic_ADR(scalingValues)
        self.use_plda = use_plda
    def getAccuracy(self, probas):
        self.bnav.eval()
        with torch.no_grad():
            olabels = probas.argmax(dim=2)

            matches = self.labels[:, self.n_lsamples:].eq(olabels).float()

            acc_test = matches.mean(1)

            m = acc_test.mean().item()
            pm = acc_test.std().item() * 1.96 / math.sqrt(self.n_runs)
            return acc_test, m, pm

    def performEpoch(self, model, epochInfo=None):
        #print(model.orig_ndatas.shape, self.probas.shape)
        probas = None
        if self.use_plda:
            probas = self.probas.detach()
            if probas.size(1) != model.orig_ndatas.size(1):
                train_labels = self.labels[:, :self.n_lsamples].long()
                gt = torch.zeros(self.n_runs, self.n_lsamples, probas.size(-1)).cuda().scatter_(2, train_labels.unsqueeze(2), 1)
                probas = torch.cat([gt, probas], dim=1)
            model.ndatas = self.adr.pLDA(model.orig_ndatas, probas, 10)
        if model.mus is None:
            model.initFromLabelledDatas(probas)
        assert model.mus is not None
        p_xj = model.getProbas()
        self.probas = p_xj

        if self.verbose:
            print("accuracy from filtered probas", self.getAccuracy(self.probas)[1])

        # self.probas = F.softmax(self.probas, 1)
        m_estimates = model.estimateFromMask(self.probas)

        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

        if self.verbose:
            pass
            #op_xj = model.getProbas()
            #acc = self.getAccuracy(op_xj)
            #print("output model accuracy", acc)

    def loop(self, model, n_epochs=20):
        # self.probas = model.getProbas()
        # model.apply(init_wights)
        #if self.verbose:
        #    print("initialisation model accuracy", self.getAccuracy(self.probas))

        if self.progressBar:
            if type(self.progressBar) == bool:
                pb = tqdm(total=n_epochs)
            else:
                pb = self.progressBar

        for epoch in tqdm(range(1, n_epochs + 1)):
            if self.verbose:
                print("----- epoch[{:3d}]  lr_p: {:0.3f} ".format(epoch, self.alpha))
            self.performEpoch(model, epochInfo=(epoch, n_epochs))
            #model.bnav.reset()
            if (self.progressBar): pb.update()

        # get final accuracy and return it
        op_xj = model.getProbas()
        acc_test, m, pm = self.getAccuracy(op_xj)
        return acc_test, m, pm


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

"""
if __name__ == '__main__':
    fix_seed(2022)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 80.68 80.71
    # epochs: 50, lr: 0.001, rho: 0.99; alpha: 0.05, beta: 0.9==>0.8093
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)  # 50
    parser.add_argument('--lr', type=float, default=0.001)  # 0.001
    parser.add_argument('--rho', type=float, default=0.2)  # 0.5 0.8
    parser.add_argument('--loss', type=str, default='ct')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--out_dir', type=str, default='out')
    parser.add_argument('--alpha', type=float, default=0.1)  # 0.05
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='cub-new')
    parser.add_argument('--backbone', type=str, default='WRN')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=128)
    args = parser.parse_args()
    alpha = args.alpha
    lr = None
    n_ways = 5
    n_queries = 15
    n_runs = 1000
    n_epochs = args.n_epochs
    n_shot = args.shot
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    b_navigator = model.BNavigator(batch_size=n_runs, hidden=args.hidden, dim=n_samples)
    b_navigator.to_device(device)
    # because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to
    # optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
    optim_bnav = optim.Adam(b_navigator.params, lr=args.lr, betas=(0.5, 0.99))
    use_gpu = torch.cuda.is_available()

    distribution = 'dirichlet'  # uniform or dirichlet

    import FSLTask_im_1

    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries, 'tasks': n_runs, 'sample': distribution}
    FSLTask_im_1.loadDataSet(args.backbone, args.dataset)
    FSLTask_im_1.setRandomStates(cfg)
    ndatas, labels, query_samples = FSLTask_im_1.GenerateRunSet(cfg=cfg)
    if cfg['sample'] == 'uniform':
        ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
        labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                            n_samples)
    elif cfg['sample'] == 'dirichlet':
        pass

    # Power transform
    beta = 0.7  # 0.9
    ndatas[:, ] = torch.pow(ndatas[:, ] + 1e-6, beta)
    ndatas = QRreduction(ndatas)
    n_nfeat = ndatas.size(2)
    ndatas = scaleEachUnitaryDatas(ndatas)
    #    ndatas = centerDatas(ndatas)
    ndatas = F.normalize(ndatas, dim=2)
    # trans-mean-sub

    print("size of the datas...", ndatas.size())  # [100,100,100]

    start = time.time()

    ndatas = ndatas.to(device)
    labels = labels.to(device)
    ndatas_l = ndatas[:, n_lsamples:, :]

    # MAP
    lam = 10
    model = GaussianModel(n_ways, lam)
    model.initFromLabelledDatas()

    print(f"shot:{n_shot}, alpha:{alpha}, lr:{args.lr}, n_epochs:{n_epochs}, args.epochs:{args.epochs}")
    optim = MAP(alpha, lr)

    optim.verbose = True
    optim.progressBar = True

    acc_test = optim.loop(model, n_epochs)

    print("final accuracy found {:0.2f} +- {:0.2f}".format(*(100 * x for x in acc_test)))
    end = time.time()
    print('time %.2fs' % (end - start))
"""


class DataSet:
    data: None
    labels: None

    def __init__(self, data=None, labels=None, n_shots=1, n_ways=5, n_queries=15):
        self.data = data
        self.labels = labels
        self.n_shots = n_shots
        self.n_ways = n_ways
        self.n_lsamples = n_ways*n_shots
        self.n_queries = n_queries
        self.n_usamples = n_ways*n_queries
        if self.data is not None:
            self.n_runs = data.size(0)
            self.n_samples = data.size(1)
            self.n_feat = data.size(2)

            if self.n_samples != self.n_lsamples + self.n_usamples:
                print("Invalid settings: queries incorrect wrt size")
                self.exit()

    def cuda(self):
        self.data = self.data.cuda()
        self.labels = self.labels.cuda()

    def cpu(self):
        self.data = self.data.cpu()
        self.labels = self.labels.cpu()

    def printState(self):
        print("DataSet: {}-shot, {}-ways, {}-queries, {}-runs, {}-feats".format( \
             self.n_shots, self.n_ways, self.n_queries, self.n_runs, self.n_feat))
        print("\t {}-labelled {}-unlabelled {}-tot".format( \
              self.n_lsamples, self.n_usamples, self.n_samples))

def load_trainset(dir_base_data):
    print(os.getcwd())
    base_output = load_pickle(dir_base_data)
    base_data = []
    num_cls = []
    for i in list(base_output.keys()):
        base_data_cls = torch.tensor(np.array(base_output[i]))
        num_cls.append(base_data_cls.shape[0])
        base_data.append(base_data_cls)

    base_data = torch.cat(base_data, dim=0) # X_base en taille [64, 600, 640]

    return base_data, num_cls

def computeSharedCov(base_data, num_cls, postRescale=False):
    # compute scov over all classes
    count = 0
    covs = []
    for i in range(len(num_cls)):
        n_cls = num_cls[i]
        base_data_c = base_data[count:count+n_cls].unsqueeze(0)
        count += n_cls
        res = base_data_c - base_data_c.mean(dim=1, keepdim=True)
        cov = torch.bmm(res.permute(0,2,1), res) # [64, 640, 640]
        covs.append(cov)

    covs = torch.cat(covs, dim=0)
    norms = covs.norm(dim=(1,2), keepdim=True)
    scov = (covs/norms).mean(0)
    sncov = norms.mean(0)

    if postRescale:
        return scov*sncov
    else:
        return scov


def getRunSet(n_shots, n_ways, n_queries, preprocess='ME', dataset='mini', model='WRN', data=None, labels=None):
    ds = DataSet(data, labels, n_shots=n_shots, n_ways=n_ways, n_queries=n_queries)
    # save_dir = './checkpoints/{}/{}'.format(dataset, model)
    # ds.dir_base_data = save_dir + '/base_{}.plk'.format(dataset)

    ds.dir_base_data = os.path.join(f'../features/{model}/{dataset}/base.plk')
    ds.base_data, ds.num_cls = load_trainset(ds.dir_base_data)
    ds.base_data = ds.base_data.to(ds.data.device)

    ds.rdata_b = ds.data.clone()

    for p in preprocess:
        if p == "M":
            print("--- preprocess: Mean subtraction")
            base_mu = ds.base_data.mean(dim=0, keepdim=True)
            ds.data = ds.data - base_mu
            ds.base_data = ds.base_data - base_mu

        elif p == "V":
            print("--- preprocess: Rotation using base data")
            Sw_base = computeSharedCov(ds.base_data, ds.num_cls, True)
            Sw_base += torch.eye(Sw_base.shape[-1]).cuda() * 1e-6

            Sw_base_U = (ds.U.permute(0,2,1).matmul(Sw_base.inverse()).matmul(ds.U)).inverse()
            e_Swb_Udata, v_Swb_Udata = torch.linalg.eigh(Sw_base_U)

            e_Swb_Udata, idx = e_Swb_Udata.sort(dim=1, descending=True)
            idx = idx.unsqueeze(1).expand(-1, ds.data.shape[2], -1)
            v_Swb_Udata = v_Swb_Udata.gather(dim=2, index=idx)

            ds.e_Swb_Udata = e_Swb_Udata
            ds.rdata_b = ds.data.matmul(v_Swb_Udata)

        elif p == "S":
            print("--- preprocess: Data projection on U")
            ds.U, ds.S, _ = torch.linalg.svd(ds.data.permute(0,2,1), full_matrices=False) # U:[n_runs, 640, N], S:[N], Vh = [N, N]
            ds.data = torch.bmm(ds.data, ds.U)

        elif p == "E":
            print("--- preprocess: Euclidean normalization")
            ds.data = scaleEachUnitaryDatas(ds.data)
            ds.base_data = ds.base_data / ds.base_data.norm(dim=1, keepdim=True)

        else:
            print("unknown preprocessing!!")
            pass

    return ds

class Basic_ADR:

    def __init__(self, scalingValues=None):
        self.scalingValues = scalingValues

    def qProjs(self, mus):
        # get projections
        dmus = mus - mus[:,:1,:]
        q, _ = torch.linalg.qr(dmus[:, 1:].permute(0, 2, 1))
        return q

    def proj(self, X, probas, beta_o):
        mus = probas.permute(0, 2, 1).matmul(X).div(probas.sum(dim=1).unsqueeze(2)+beta_o)
        q = self.qProjs(mus)
        pX = X.matmul(q)

        return pX

    def pLDA(self, X, probas, beta_o):

        if self.scalingValues is None:
            return self.proj(X, probas)

        # spherize datas
        Y = X.mul(self.scalingValues.unsqueeze(1))
        # get centroids
        Ymus = probas.permute(0,2,1).matmul(Y).div(probas.sum(dim=1).unsqueeze(2)+beta_o)

        q = self.qProjs(Ymus)
        # get projected signal to be used
        pX = Y.matmul(q)

        return pX


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
