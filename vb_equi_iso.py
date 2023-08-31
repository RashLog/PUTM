import argparse
import math
import os
import pickle as pkl
import sys
from time import time

import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
import matplotlib.pyplot as plt
import numpy as np

#-------- class to handle a Symmetric Dirichlet mixture model

def softmax(x, dim=2): # [n_runs, N, n_ways]
    eps = torch.finfo(torch.float32).eps
    maxes = torch.max(x, dim, keepdim=True)[0]
    x_exp = torch.exp(x - maxes) # [n_runs, N, n_ways]
    x_exp_sum = torch.sum(x_exp, 2, keepdim=True) # [n_runs, N, 1]
    out = (x_exp+eps)/(x_exp_sum+eps)
    
    return out


def log_(x):
    eps = torch.finfo(torch.float32).eps
    
    return torch.log(x + eps)


def update_beta_k(rnk, beta_o): # [n_runs, N, n_ways]
    """
    Update beta_k
    """
    return beta_o + rnk.sum(1)


def update_mu_k(beta_k, rnk, mu_o, beta_o, X): # [n_runs, n_ways]
    """
    Update mu_k
    """
    d1 = (1./beta_k).unsqueeze(2) # [n_runs, n_ways, 1]
    d2 = mu_o * beta_o + torch.bmm(rnk.permute(0,2,1), X) # [n_runs, n_ways, N] x [n_runs, N, D]
    mu_k = d1 * d2
   
    return mu_k 


class VB_SymDirMM:
    
    def __init__(self, ds, alpha0=2, fixedModel=False):
        self.ds = ds
        self.alpha0 = alpha0
        self.alpha_k = torch.Tensor(self.ds.n_runs, self.ds.n_ways).fill_(alpha0)
        self.fixed = fixedModel
        
    def update(self, rnk):
        if not self.fixed:
            self.alpha_k = rnk.sum(dim=1) + self.alpha0 - self.ds.n_shots
        return self.alpha_k
        
    def getLnPiks(self):
        if self.fixed:
            return torch.Tensor(1,1).fill_(0).to(self.ds.data.device)
        else: 
            return (self.alpha_k.digamma() - self.alpha_k.sum(dim=1, keepdim=True).digamma())
        
#--------------------------------------------------
#------------ Classes for handling Adaptive Dimension Reduction ADR
    

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

#-----------   Class to handle ISO GaussianModel

class VB_IsoGM:
    
    def __init__(self, ds, eValues=None, sigma2=0.02):
        
        self.ds = ds
        self.beta_o = 10
        self.mu_o = 0
        self.ADR = Basic_ADR(eValues)
        self.wDim = self.ds.n_ways-1
        self.sigma2 = sigma2
                    
    def update(self, X, rnk):
        
        self.X_proj = self.ADR.pLDA(X, rnk, self.beta_o)
        self.beta_k = update_beta_k(rnk, self.beta_o)
        self.mu_k_proj = update_mu_k(self.beta_k, rnk, self.mu_o, self.beta_o, self.X_proj)
        return self.X_proj, self.beta_k, self.mu_k_proj
        
    def getLnPnk(self):
        
        dist2 = (self.X_proj.unsqueeze(2) - self.mu_k_proj.unsqueeze(1)).norm(dim=3).pow(2)
        scores = - dist2 / (2*self.sigma2)
        scores = scores - (self.wDim / 2.) * torch.tensor(2*math.pi).log()
        scores = scores - self.wDim / (2 * self.beta_k.unsqueeze(1))
            
        return scores


class VB_EM:
    
    def __init__(self, ds, mixtureModel=None, clusterModel=None):
        self.ds = ds
        self.mixtureModel = mixtureModel
        self.clusterModel = clusterModel
        
    def E_step(self):
        lnPiks = self.mixtureModel.getLnPiks()
        lnPnk = self.clusterModel.getLnPnk()
        lnRonk =  lnPiks.unsqueeze(1) + lnPnk   # dim [nruns][nsamples][nways]
        
        rnk = softmax(lnRonk, dim=2)
        train_labels = self.ds.labels[:,:self.ds.n_lsamples].long()
        rnk[:,:self.ds.n_lsamples] = 0
        rnk[:,:self.ds.n_lsamples].scatter_(2, train_labels.unsqueeze(2), 1)
        
        return rnk
    
    def M_step(self, X, rnk):
        alpha_k = self.mixtureModel.update(rnk)
        X_proj, beta_k, mu_k_proj = self.clusterModel.update(X, rnk)
        return alpha_k, X_proj, beta_k, mu_k_proj
        
    def loop(self, X, rnk, nIter=20):
        
        for iter in range(nIter):
            self.M_step(X, rnk)
            rnk = self.E_step()
            
        return rnk