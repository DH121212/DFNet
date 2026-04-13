import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, sr, hr):
        sr_gray = sr.mean(dim=1, keepdim=True)
        hr_gray = hr.mean(dim=1, keepdim=True)

        sr_fft = torch.fft.rfftn(sr_gray, s=(sr_gray.shape[-2], sr_gray.shape[-1]))
        hr_fft = torch.fft.rfftn(hr_gray, s=(hr_gray.shape[-2], hr_gray.shape[-1]))

        loss_fft = self.loss(sr_fft, hr_fft)

        return loss_fft

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0,  0,  0],
                                [1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.loss = nn.L1Loss()

    def forward(self, sr, hr):
        sr_gray = sr.mean(dim=1, keepdim=True)
        hr_gray = hr.mean(dim=1, keepdim=True)

        grad_sr_x = F.conv2d(sr_gray, self.sobel_x, padding=1)
        grad_sr_y = F.conv2d(sr_gray, self.sobel_y, padding=1)
        grad_hr_x = F.conv2d(hr_gray, self.sobel_x, padding=1)
        grad_hr_y = F.conv2d(hr_gray, self.sobel_y, padding=1)

        return self.loss(grad_sr_x, grad_hr_x) + self.loss(grad_sr_y, grad_hr_y)

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()

        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')

            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss(reduction='mean')
            elif loss_type == 'Gradient':
                loss_function = GradientLoss()

            elif loss_type == 'FFT':
                loss_function = FFTLoss()
            else:
                assert False, f"Unsupported loss type: {loss_type:s}"
            
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.pdf'.format(apath, l['type']))
            plt.close(fig)

    def save(self, apath):
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))
