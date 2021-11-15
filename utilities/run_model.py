import torch
import time
from tqdm import tqdm
from .constants import *
from utilities.device import get_device
from .lr_scheduling import get_lr

from dataset.e_piano import compute_epiano_accuracy
from torch.autograd import Variable

# train_epoch
def train_epoch(cur_epoch, model, dataloader, loss, opt, lr_scheduler=None, print_modulus=1, weights=None):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Trains a single model epoch
    ----------
    """

    out = -1
    model.train()
    for batch_num, batch in tqdm(enumerate(dataloader)):
        time_before = time.time()

        opt.zero_grad()

        x   = batch[0].to(get_device())
        tgt = batch[1].to(get_device())

        y = model(x)

        y   = y.reshape(y.shape[0] * y.shape[1], -1)
        tgt = tgt.flatten()

        out = loss.forward(y, tgt)
        if weights is not None:
            out = out*weights
        out = torch.mean(out)

        out.backward()
        opt.step()

        if(lr_scheduler is not None):
            lr_scheduler.step()

        time_after = time.time()
        time_took = time_after - time_before

        if((batch_num+1) % print_modulus == 0):
            print(SEPERATOR)
            print("Epoch", cur_epoch, " Batch", batch_num+1, "/", len(dataloader))
            print("LR:", get_lr(opt))
            print("Train loss:", float(out))
            print("")
            print("Time (s):", time_took)
            print(SEPERATOR)
            print("")

    return

# train_epoch
def train_GAN(cur_epoch, G, D, G_opt, D_opt, dataloader, loss, scheduler_G=None, scheduler_D=None, print_modulus=1, n_critics=3):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Trains a single model epoch
    ----------
    """

    out = -1
    G.train()
    D.train()
    for batch_num, batch in enumerate(dataloader):
        time_before = time.time()

        x   = batch[0].to(get_device())
        tgt = Variable(batch[1]).to(get_device())
        # valid = Variable(torch.Tensor(x.shape[0],1).fill_(1.0), requires_grad=False).to(get_device())
        # fake = Variable(torch.Tensor(x.shape[0],1).fill_(0.0), requires_grad=False).to(get_device())

        # Train discriminator
        if batch_num % n_critics == 0:
            D_opt.zero_grad()

            y = G(x)
            real_valid = D(tgt)
            fake_valid = D(torch.argmax(y, dim=-1).detach())
            loss_D = -torch.mean(real_valid) + torch.mean(fake_valid)

            loss_D.backward()
            D_opt.step()

            if (scheduler_D is not None):
                scheduler_D.step()
        
        G_opt.zero_grad()

        x = x.detach()
        tgt = tgt.detach()

        y = G(x)
        # _, real_f = D(tgt, w_feat=True)
        fake_val = D(torch.argmax(y, dim=-1))
        loss_adv = -torch.mean(fake_val)

        # loss_fm = torch.sqrt(torch.mean((fake_f - real_f.detach()).pow(2)))

        y   = y.reshape(y.shape[0] * y.shape[1], -1)
        out = loss.forward(y, tgt.flatten())

        out = out+0.1*loss_adv

        out.backward()
        G_opt.step()

        if(scheduler_G is not None):
            scheduler_G.step()

        time_after = time.time()
        time_took = time_after - time_before

        print('\rEpoch %d Batch %d/%d Train loss: G %.4f D %.4f' % (cur_epoch, batch_num+1, len(dataloader), float(out), float(loss_D)), end='')
        # if((batch_num+1) % print_modulus == 0):
        #     print(SEPERATOR)
        #     print("Epoch", cur_epoch, " Batch", batch_num+1, "/", len(dataloader))
        #     print("LR:", get_lr(G_opt))
        #     print("Train loss: G %.4f D %.4f" %  (float(out), float(loss_D)))
        #     print("")
        #     print("Time (s):", time_took)
        #     print(SEPERATOR)
        #     print("")

    print()
    return

# eval_model
def eval_model(model, dataloader, loss):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Evaluates the model and prints the average loss and accuracy
    ----------
    """

    model.eval()

    avg_acc     = -1
    avg_loss    = -1
    with torch.set_grad_enabled(False):
        n_test      = len(dataloader)
        sum_loss   = 0.0
        sum_acc    = 0.0
        for batch in tqdm(dataloader):
            x   = batch[0].to(get_device())
            tgt = batch[1].to(get_device())

            y = model(x)

            sum_acc += float(compute_epiano_accuracy(y, tgt))

            y   = y.reshape(y.shape[0] * y.shape[1], -1)
            tgt = tgt.flatten()

            out = loss.forward(y, tgt)

            sum_loss += float(out)

        avg_loss    = sum_loss / n_test
        avg_acc     = sum_acc / n_test

    return avg_loss, avg_acc
