import torch
import datetime
import numpy as np
from torch.autograd import Variable

from lib.utility_general import flatten_grad
import torch.nn as nn
from torch.autograd import grad
import random # to shuffle the data
from tqdm.notebook import trange # for the progress bar

np.set_printoptions(threshold=10010)

def grad_z(z, t, model, loss_criterion, n, λ=0):
    model.eval()
    # initialize
    z, t = Variable(z), Variable(t) # here were two flags: volatile=False. True would mean that autograd shouldn't follow this. Got disabled
    y = model(z)
    
    loss = loss_criterion(y, t)

    # We manually add L2 regularization
    l2_reg = 0.0
    for param in model.parameters():
        l2_reg += torch.norm(param)**2
    loss += 1/n * λ/2 * l2_reg

    return list(grad(loss, list(model.parameters()), create_graph=True))

def find_hessian(loss, model):
    grad1 = torch.autograd.grad(loss, model.parameters(), create_graph=True) #create graph important for the gradients

    grad1 = flatten_grad(grad1)
    list_length = grad1.size(0)
    hessian = torch.zeros(list_length, list_length)

    for idx in trange(list_length, desc="Calculating hessian", position=0, leave=True):
        grad2rd = torch.autograd.grad(grad1[idx], model.parameters(), create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2.detach().cpu()
        del g2

    H = hessian.cpu().data.numpy()
    # calculate every element separately -> detach after calculating all 2ndgrad from this 1grad
    return H

def find_heigenvalues(loss, model):
    H = find_hessian(loss, model)
    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues

def biggest_heigenvalue(hessian_or_loss, model = None, negative=None):
    if model is None:
        eigenvalues = np.linalg.eigvalsh(hessian_or_loss)
    else:
        eigenvalues = find_heigenvalues(hessian_or_loss, model)

    eigenvalues = np.sort(eigenvalues)
    big = eigenvalues[-1]
    small = eigenvalues[0]

    if negative is None:
        if abs(small) > abs(big):
            return small
        else:
            return big
    else:
        if small < 0.0:
            return small
        else:
            print("Hessian has no negative eigenvalue!")
            return None