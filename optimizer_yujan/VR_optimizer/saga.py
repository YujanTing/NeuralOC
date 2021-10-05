import torch
from torch.optim import Optimizer
import numpy as np


class SAGA(Optimizer):

    def __int__(self, params, lr=0.01, n=1024, seed=77):
        tmp = dict(lr=lr, n=n, seed=seed)
        super(SAGA, self).__init__(params, tmp)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                n = group['n']
                seed = group['seed']
                lr = group['lr']

                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # initialize state
                if len(state) == 0:
                    state['step'] = 0
                    # Sum of the gradients
                    state['sum_gradient'] = torch.zeros_like(p.data)
                    # Store the vector v^j for each j
                    if len(p.data.size()) == 1:
                        state['vj_memory'] = torch.zeros(n, p.data.size()[0])
                    elif len(p.data.size()) == 2:
                        state['vj_memory'] = torch.zeros(n, p.data.size()[0], p.data.size()[1])
                    else:
                        print('Length error')
                    state['rd'] = np.random.RandomState(seed)

                sum_gradient = state['sum_gradient']
                vj_memory = state['vj_memory']
                rd = state['rd']

                state['step'] += 1

                # sample i_k from  {1,...,n}
                i_k = int(rd.rand(1) * n)

                # get v_old
                v_old = vj_memory[i_k]

                # print('v_old before')
                # print(v_old )
                # print('----------------')

                # update parameter
                p.data.add_(grad - v_old + sum_gradient, alpha=(-1)*lr)

                # update the sum of gradient g_k
                sum_gradient.add_(grad, alpha=(1 / n)).add_(v_old, alpha=(-1 / n))

                # update v_ik
                vj_memory[i_k] = grad

                # print('v_old after')
                # print(v_old)
                # print('-----------')
                # print('vj_memory[i_k]')
                # print(vj_memory[i_k])