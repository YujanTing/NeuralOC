import torch
from torch.optim import Optimizer

class SAGA(Optimizer):

    def __int__(self, params, lr, n):
        tmp = dict(lr=lr, n=n)
        super(SAGA, self).__init__(params, tmp)

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

                # get v_old
                v_old = v

                # update parameter
                p.data.add_(grad - v_old + g, alpha=(-1)*lr)

                # update the g
                g.add_(grad, alpha=(1 / n)).add_(v_old, alpha=(-1 / n))

                # update v
                v = grad