'''MIT License

Copyright (c) 2019 Yueqi Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import torch
from torch.cuda import device
from src.OCflow import OCflow

def train_epoch_SVRG(itr, net, net_snapshot, optimizer_k, optimizer_snapshot, train_loader, prob, tspan, nt, stepper, alph):

    net.train()
    net_snapshot.train()
    loss = AverageCalculator()

    # calculate the mean gradient
    optimizer_snapshot.zero_grad()  # zero_grad outside for loop, accumulate gradient inside
    for x in train_loader:
        Jc_snapshot, cs_snapshot = OCflow(x, net_snapshot, prob, tspan, nt, stepper, alph)
        Jc_snapshot /= len(train_loader)
        Jc_snapshot.backward()

    # pass the current paramesters of optimizer_0 to optimizer_k
    u = optimizer_snapshot.get_param_groups()
    optimizer_k.set_u(u)

    for x in train_loader:
        Jc_iter, cs_iter = OCflow(x, net, prob, tspan, nt, stepper, alph)

        # optimization
        optimizer_k.zero_grad()
        Jc_iter.backward()

        Jc_snapshot2, cs_snapshot2 = OCflow(x, net_snapshot, prob, tspan, nt, stepper, alph)

        optimizer_snapshot.zero_grad()
        Jc_snapshot2.backward()

        optimizer_k.step(optimizer_snapshot.get_param_groups())

        # logging
        loss.update(Jc_iter.data.item())
        itr += 1
        print('itr: ' + str(itr))
        print('loss.avg' + str(loss.avg))

    # update the snapshot
    optimizer_snapshot.set_param_groups(optimizer_k.get_param_groups())

    return loss.avg, cs_iter, itr, net, net_snapshot

def validate_epoch(net, val_loader, prob, tspan, nt, stepper, alph):
    """One epoch of validation
    """
    net.eval()
    loss = AverageCalculator()

    for x in val_loader:
        Jc, cs = OCflow(x, net, prob, tspan, nt, stepper, alph)

        # logging
        loss.update(Jc.data.item())

    return loss.avg, cs, net

# aux functions
def accuracy(yhat, labels):
    _, indices = yhat.max(1)
    return (indices == labels).sum().data.item() / float(len(labels))

class AverageCalculator():
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, val, n=1):
        assert(n > 0)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / float(self.count)
