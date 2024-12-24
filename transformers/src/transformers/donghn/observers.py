import torch
from torch import nn

class ObserverBase(nn.Module):
    def __init__(self, channels):
        super(ObserverBase, self).__init__()
        self.channels = channels
        if self.channels > 0:  # only weights are used
            self.register_buffer('max_val', torch.zeros((channels, 1), dtype=torch.float32))
            self.register_buffer('min_val', torch.zeros((channels, 1), dtype=torch.float32))
        else:
            self.register_buffer('max_val', torch.zeros((1), dtype=torch.float32))
            self.register_buffer('min_val', torch.zeros((1), dtype=torch.float32))
        self.num_flag = 0

    def reset_range(self):
        self.min_val = torch.zeros_like(self.min_val) + 10000
        self.max_val = torch.zeros_like(self.max_val) - 10000

'''
class NineNineObserver(ObserverBase):
    def __init__(self, channels, nine=0.999):
        super(NineNineObserver, self).__init__(channels)
        self.nine = nine

    def forward(self, x):
        if self.channels > 0:
            nine_nine_id = int(x.shape[1] * self.nine)
            indata, _ = torch.sort(indata, dim=1)
            min_val = indata[:, 0]
            max_val = indata[:, nine_nine_id]
        else:
            indata = torch.flatten(x)
            nine_nine_id = int(indata.shape[0] * self.nine)
            indata, _ = torch.sort(indata)

            min_val = indata[0]
            max_val = indata[nine_nine_id]
        self.update_range(min_val, max_val)

    def update_range(self, min_val, max_val):
        min_val = torch.reshape(min_val, self.min_val.shape)
        max_val = torch.reshape(max_val, self.max_val.shape)

        if self.num_flag == 0:
            min_val_new = min_val
            max_val_new = max_val
            self.num_flag += 1
        else:
            min_val_new = torch.min(min_val, self.min_val)
            max_val_new = torch.max(max_val, self.min_val)

        self.min_val.copy_(min_val_new.detach())
        self.max_val.copy_(max_val_new.detach())
'''


class NineNineObserver(ObserverBase):
    def __init__(self, channels, nine=0.6):
        super(NineNineObserver, self).__init__(channels)
        self.nine = nine

    def forward(self, x):
        if self.channels > 0:
            nine_nine_id = int(x.shape[1] * self.nine)
            ###indata, _ = torch.sort(indata, dim=1)
            ###indata, _ = torch.sort(x, dim=1)  # 'indata'를 'x'로 변경
            indata, _ = torch.sort(torch.abs(x), dim=1)  # 절댓값 기준으로 정렬
            min_val = indata[:, 0]
            max_val = indata[:, nine_nine_id]
        else:
            ###indata = torch.flatten(x)
            indata = torch.flatten(torch.abs(x))  # 절댓값을 구한 후 평탄화
            nine_nine_id = int(indata.shape[0] * self.nine)
            indata, _ = torch.sort(indata)

            min_val = indata[0]
            max_val = indata[nine_nine_id]
        self.update_range(min_val, max_val)

    def update_range(self, min_val, max_val):
        min_val = torch.reshape(min_val, self.min_val.shape)
        max_val = torch.reshape(max_val, self.max_val.shape)

        if self.num_flag == 0:
            min_val_new = min_val
            max_val_new = max_val
            self.num_flag += 1
        else:
            min_val_new = torch.min(min_val, self.min_val)
            max_val_new = torch.max(max_val, self.min_val)

        self.min_val.copy_(min_val_new.detach())
        self.max_val.copy_(max_val_new.detach())


class MinMaxObserver(ObserverBase):
    def __init__(self, channels):
        super(MinMaxObserver, self).__init__(channels)

    def forward(self, x):
        if self.channels > 0:
            min_val = torch.min(x, 1, keepdim=True)[0]
            max_val = torch.max(x, 1, keepdim=True)[0]
        else:
            min_val = torch.min(x)
            max_val = torch.max(x)
        self.update_range(min_val, max_val)


class NormalMinMaxObserver(MinMaxObserver):
    def __init__(self, channels):
        super(NormalMinMaxObserver, self).__init__(channels)

    def update_range(self, min_val, max_val):
        min_val = torch.reshape(min_val, self.min_val.shape)
        max_val = torch.reshape(max_val, self.max_val.shape)

        if self.num_flag == 0:
            min_val_new = min_val
            max_val_new = max_val
            self.num_flag += 1
        else:
            min_val_new = torch.min(min_val, self.min_val)
            ###max_val_new = torch.max(max_val, self.min_val) # original code
            max_val_new = torch.max(max_val, self.max_val) # my code

        self.min_val.copy_(min_val_new.detach())
        self.max_val.copy_(max_val_new.detach())

class MovingAvgMinMaxObserver(MinMaxObserver):
    def __init__(self, channels, momentum=0.1):
        super(MovingAvgMinMaxObserver, self).__init__(channels)
        self.momentum = momentum

    def update_range(self, min_val, max_val):
        min_val = torch.reshape(min_val, self.min_val.shape)
        max_val = torch.reshape(max_val, self.max_val.shape)

        if self.num_flag == 0:
            min_val_new = min_val
            max_val_new = max_val
            self.num_flag += 1
        else:
            min_val_new = self.min_val + (min_val - self.min_val) * self.momentum
            max_val_new = self.max_val + (max_val - self.max_val) * self.momentum

        self.min_val.copy_(min_val_new.detach())
        self.max_val.copy_(max_val_new.detach())