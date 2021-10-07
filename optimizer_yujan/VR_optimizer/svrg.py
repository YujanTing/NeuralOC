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

from torch.optim import Optimizer
import copy


class SVRG_k(Optimizer):
    r"""Optimization class for calculating the gradient of one iteration.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """
    def __init__(self, params, lr, weight_decay=0):
        print("Using optimizer: SVRG")
        self.u = None
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SVRG_k, self).__init__(params, defaults)

    def get_param_groups(self):
        return self.param_groups

    def set_u(self, new_u):
        """Set the mean gradient for the current epoch.
        """
        if self.u is None:
            self.u = copy.deepcopy(new_u)
        for u_group, new_group in zip(self.u, new_u):
            for u, new_u in zip(u_group['params'], new_group['params']):
                u.grad = new_u.grad.clone()

    def step(self, params):
        """Performs a single optimization step.
        """
        for group, new_group, u_group in zip(self.param_groups, params, self.u):
            weight_decay = group['weight_decay']
            lr = group['lr']

            for p, q, u in zip(group['params'], new_group['params'], u_group['params']):
                if p.grad is None:
                    continue
                if q.grad is None:
                    continue
                # core SVRG gradient update
                new_d = p.grad.data - q.grad.data + u.grad.data
                if weight_decay != 0:
                    new_d.add_(weight_decay, p.data)
                p.data.add_(new_d, alpha=(-1)*lr)


class SVRG_Snapshot(Optimizer):
    r"""Optimization class for calculating the mean gradient (snapshot) of all samples.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """
    def __init__(self, params):
        defaults = dict()
        super(SVRG_Snapshot, self).__init__(params, defaults)

    def get_param_groups(self):
        return self.param_groups

    def set_param_groups(self, new_params):
        """Copies the parameters from another optimizer.
        """
        for group, new_group in zip(self.param_groups, new_params):
            for p, q in zip(group['params'], new_group['params']):
                p.data[:] = q.data[:]