import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable

# Borrowed from https://github.com/jason9693/MusicTransformer-pytorch/blob/5f183374833ff6b7e17f3a24e3594dedd93a5fe5/custom/criterion.py#L28
class SmoothCrossEntropyLoss(_Loss):
    """
    https://arxiv.org/abs/1512.00567
    """
    __constants__ = ['label_smoothing', 'vocab_size', 'ignore_index', 'reduction']

    def __init__(self, label_smoothing, vocab_size, ignore_index=-100, reduction='mean', is_logits=True):
        assert 0.0 <= label_smoothing <= 1.0
        super().__init__(reduction=reduction)

        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.input_is_logits = is_logits

    def forward(self, input, target):
        """
        Args:
            input: [B * T, V]
            target: [B * T]
        Returns:
            cross entropy: [1]
        """
        mask = (target == self.ignore_index).unsqueeze(-1)
        q = F.one_hot(target.long(), self.vocab_size).type(torch.float32)
        u = 1.0 / self.vocab_size
        q_prime = (1.0 - self.label_smoothing) * q + self.label_smoothing * u
        q_prime = q_prime.masked_fill(mask, 0)

        ce = self.cross_entropy_with_logits(q_prime, input)
        if self.reduction == 'mean':
            lengths = torch.sum(target != self.ignore_index)
            return ce.sum() / lengths
        elif self.reduction == 'sum':
            return ce.sum()
        else:
            raise NotImplementedError

    def cross_entropy_with_logits(self, p, q):
        return -torch.sum(p * (q - q.logsumexp(dim=-1, keepdim=True)), dim=-1)


class UncertaintyLoss(nn.Module):
    
    def __init__(self, v_num):
        super(UncertaintyLoss, self).__init__()
        sigma = torch.randn(v_num)
        self.sigma = nn.Parameter(sigma)
        self.v_num = v_num

    def forward(self, *input):
        loss = 0
        for i in range(self.v_num):
            loss += input[i] / (2 * self.sigma[i] ** 2)
        loss += torch.log(self.sigma.pow(2).prod())
        return loss

class WGAN_DIV_Loss(nn.Module):
    def __init__(self, k=2, p=6):
        super(WGAN_DIV_Loss, self).__init__()
        self.k = k
        self.p = p
    
    def forward(self, real_val, y, fake_val, y_hat):
        real_grad_out = Variable(torch.cuda.FloatTensor(real_val.size(0), 1).fill_(1.0), requires_grad=False)
        real_grad = torch.autograd.grad(
            real_val, y, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (self.p/2)

        fake_grad_out = Variable(torch.cuda.FloatTensor(fake_val.size(0), 1).fill_(0.0), requires_grad=False)
        fake_grad = torch.autograd.grad(
            fake_val, y_hat, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (self.p/2)

        div_gp = torch.mean(real_grad_norm + fake_grad_norm) * self.k / 2

        # WGAN-DIV loss
        loss_D = -torch.mean(real_val) + torch.mean(fake_val) + div_gp
        # Spectral normalization loss
        # loss_D = torch.mean(F.relu(-1 + real_val)) + torch.mean(F.relu(-1-fake_val))
    
        return loss_D