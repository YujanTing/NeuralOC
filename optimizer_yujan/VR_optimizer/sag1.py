import torch
from torch.optim import Optimizer

class SAG(Optimizer):

    def __int__(self, params, lr, n):
        defaults = dict(lr=lr, n=n)
        super(SAG, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                n = group['n']
                lr = group['lr']

                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['g'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                g = state['g']
                v = state['v']

                state['step'] += 1

                # update g first time
                g.add_(v, alpha=(-1 / n))

                # update v
                v = grad

                # update g second time
                g.add_(v, alpha=1 / n)

                # update the parameter
                p.data.add_(g, alpha=(-1)*lr)


