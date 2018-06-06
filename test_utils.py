import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

def log_mean_exp(mtx):
    sh, _ = mtx.max(dim=1, keepdim=True)
    mtx = (mtx - sh).exp().mean(dim=1).log() + sh.view(-1)
    return mtx

def gaussian_log_pdf(distr, samples):
    mu, sigma = distr
    K = samples.shape[0] // mu.shape[0]
    n = mu.shape[0]
    d = mu.shape[1]
    samples = samples.view(n, K, d)
    mu = mu.view(n, 1, d)
    sigma = sigma.view(n, 1, d)
    loss = -sigma.log() - np.log(2 * np.pi) - (((samples - mu) / sigma) ** 2) / 2
    return loss.sum(dim=-1).view(-1)

def model_test_loss(test_data, generate_mask, compute_loss, batch_size=128, is_cuda=False, max_batches=None,
                    verbose=False, verbose_update_freq=10, num_workers=0):
    dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=(max_batches is None), num_workers=num_workers)
    num_batches = len(dataloader)
    avg_loss = 0
    for i, batch in enumerate(dataloader):
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = batch[0]
        batch = batch.view(batch.shape[0], -1)
        b = Variable(generate_mask(batch.size(0)))
        if is_cuda:
            batch = batch.cuda()
            b = b.cuda()
        loss = compute_loss(batch, b)
        avg_loss += (loss - avg_loss) / (i + 1)
        if verbose and (i + 1) % verbose_update_freq == 0:
            print('\rTest loss:', avg_loss,
                  'Batch', i + 1, 'of', num_batches, ' ' * 10, end='', flush=True)
        if verbose and (i + 1) % 100 == 0:
            print(flush=True)
        if max_batches and i >= max_batches:
            break
    return avg_loss

def compute_log_likelihood_monte_carlo(batch, b, model, loss, K):
    model.eval()
    prior = model.prior_distr(Variable(batch), b)
    z = model.sample_latent(prior, K)
    x_distr = model.generative_network(z)

    loss = loss(Variable(batch), x_distr, b)
    loss = log_mean_exp(loss)
    loss = float(loss.mean())
    del x_distr, z
    return loss

def compute_log_likelihood_importance_sampling(batch, b, model, loss, K):
    model.eval()
    proposal = model.proposal_distr(Variable(batch), b)
    prior = model.prior_distr(Variable(batch), b)
    z = model.sample_latent(proposal, K)
    x_distr = model.generative_network(z)

    loss = loss(Variable(batch), x_distr, b)
    loss += (gaussian_log_pdf(prior, z) - gaussian_log_pdf(proposal, z)).view(batch.shape[0], -1)
    loss = log_mean_exp(loss)
    loss = float(loss.mean())
    del x_distr, z
    return loss