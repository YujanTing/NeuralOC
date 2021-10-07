import torch

if __name__ == '__main__':
    a = torch.zeros(1)
    a.data += 1
    print(a)