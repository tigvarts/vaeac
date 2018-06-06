import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

def kl(q_distr, p_distr):
    p_mu, p_sigma = p_distr
    q_mu, q_sigma = q_distr
    return (p_sigma.log() - q_sigma.log() - 0.5 + (q_sigma ** 2 + (q_mu - p_mu) ** 2) / (2 * p_sigma ** 2)).sum(-1)

class UCM(nn.Module):
    def __init__(self, alpha, loss,
                 proposal_network, proposal_mu_head, proposal_sigma_head,
                 prior_network, prior_mu_head, prior_sigma_head,
                 generative_network,
                 mu_0=1e4, sigma_0=1e-4):
        super(type(self), self).__init__()
        self.alpha = alpha
        self.loss = loss
        self.proposal_network = proposal_network
        self.proposal_mu_head = proposal_mu_head
        self.proposal_sigma_head = proposal_sigma_head
        self.prior_network = prior_network
        self.prior_mu_head = prior_mu_head
        self.prior_sigma_head = prior_sigma_head
        self.generative_network = generative_network
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0

    def proposal_distr(self, x, b):
        y = x.clone()
        y[torch.isnan(y)] = 0
        data = torch.cat([y, b], dim=-1)
        data = self.proposal_network(data)
        mu = self.proposal_mu_head(data)
        raw_sigma = self.proposal_sigma_head(data)
        sigma = nn.functional.softplus(raw_sigma).clamp(min=1e-5)
        return mu, sigma

    def prior_distr(self, x, b):
        y = x.clone()
        y[torch.isnan(y)] = 0
        data = torch.cat([y * (1 - b), b], dim=-1)
        data = self.prior_network(data)
        mu = self.prior_mu_head(data)
        raw_sigma = self.prior_sigma_head(data)
        sigma = nn.functional.softplus(raw_sigma).clamp(min=1e-5)
        return mu, sigma

    def sample_latent(self, distr, K=1):
        mu, sigma = distr
        n, d = mu.shape
        e = Variable(torch.randn(n, K, d))
        if next(self.parameters()).is_cuda:
            e = e.cuda()
        mu = mu.view(n, 1, d)
        sigma = sigma.view(n, 1, d)
        return (sigma * e + mu).view(-1, d)

    def batch_loss(self, batch, weights=None):
        x, b = batch

        prior_z_distr = self.prior_distr(x, b)
        p_mu, p_sigma = prior_z_distr
        p_mu_regularizer = -(p_mu ** 2).sum(1) / 2 / (self.mu_0 ** 2)
        p_sigma_regularizer = (p_sigma.log() - p_sigma).sum(1) * self.sigma_0
        prior_regularizer = p_mu_regularizer + p_sigma_regularizer

        if self.alpha != 0:
            proposal_z_distr = self.proposal_distr(x, b)
            proposal_z_samples = self.sample_latent(proposal_z_distr, 1)
            x_distr = self.generative_network(proposal_z_samples)
            kl_loss = kl(proposal_z_distr, prior_z_distr)
            reconstruction_loss = self.loss(x, x_distr, b)
            ucm_vlb = reconstruction_loss - kl_loss
        else:
            ucm_vlb = 0

        if self.alpha != 1:
            prior_z_samples = self.sample_latent(prior_z_distr, 1)
            x_distr = self.generative_network(prior_z_samples)
            gsnn_ll = self.loss(x, x_distr, b)
        else:
            gsnn_ll = 0

        loss = self.alpha * ucm_vlb + (1 - self.alpha) * gsnn_ll + prior_regularizer
        if weights is not None:
            loss = loss * weights
        return loss.mean()

    def generate_samples(self, x, b, K):
        n, d = x.shape
        z_distr = self.prior_distr(Variable(x), Variable(b))
        z_samples = self.sample_latent(z_distr, K)
        return self.generative_network(z_samples).data.view(n, K, -1)