from torch import nn
from torch.optim import Adam

from mask_generators import MCARGenerator
from nn_utils import ResBlock, MemoryLayer, SkipConnection
from prob_utils import CategoricalToOneHotLayer, GaussianCategoricalLoss, \
                       GaussianCategoricalSampler, SetGaussianSigmasToOne


def get_imputation_networks(one_hot_max_sizes):
    """
    This function builds neural networks for imputation given
    the list of one-hot max sizes of the dataset features.
    It returns a dictionary with those neural networks together with
    reconstruction log probability function, optimizer constructor,
    sampler from the generator output, mask generator, batch size,
    and scale factor for the stability of the variational lower bound
    optimization.
    """

    width = 256
    depth = 10
    latent_dim = 64

    # Proposal network
    proposal_layers = [
        CategoricalToOneHotLayer(one_hot_max_sizes +
                                 [0] * len(one_hot_max_sizes),
                                 list(range(len(one_hot_max_sizes)))),
        nn.Linear(sum(max(1, x) for x in one_hot_max_sizes) +
                  len(one_hot_max_sizes) * 2,
                  width),
        nn.LeakyReLU(),
    ]
    for i in range(depth):
        proposal_layers.append(
            SkipConnection(
                nn.Linear(width, width),
                nn.LeakyReLU(),
            )
        )
    proposal_layers.append(
        nn.Linear(width, latent_dim * 2)
    )
    proposal_network = nn.Sequential(*proposal_layers)

    # Prior network
    prior_layers = [
        CategoricalToOneHotLayer(one_hot_max_sizes +
                                 [0] * len(one_hot_max_sizes)),
        MemoryLayer('#input'),
        nn.Linear(sum(max(1, x) for x in one_hot_max_sizes) +
                  len(one_hot_max_sizes),
                  width),
        nn.LeakyReLU(),
    ]
    for i in range(depth):
        prior_layers.append(
            SkipConnection(
                # skip-connection from prior network to generative network
                MemoryLayer('#%d' % i),
                nn.Linear(width, width),
                nn.LeakyReLU(),
            )
        )
    prior_layers.extend([
        MemoryLayer('#%d' % depth),
        nn.Linear(width, latent_dim * 2),
    ])
    prior_network = nn.Sequential(*prior_layers)

    # Generative network
    generative_layers = [
        nn.Linear(64, 256),
        nn.LeakyReLU(),
    ]
    for i in range(depth + 1):
        generative_layers.append(
            SkipConnection(
                # skip-connection from prior network to generative network
                MemoryLayer('#%d' % (depth - i), True),
                nn.Linear(width * 2, width),
                nn.LeakyReLU(),
            )
        )
    generative_layers.extend([
        MemoryLayer('#input', True),
        nn.Linear(width + sum(max(1, x) for x in one_hot_max_sizes) +
                  len(one_hot_max_sizes),
                  sum(max(2, x) for x in one_hot_max_sizes)),
        SetGaussianSigmasToOne(one_hot_max_sizes),
    ])
    generative_network = nn.Sequential(*generative_layers)

    return {
        'batch_size': 64,

        'reconstruction_log_prob': GaussianCategoricalLoss(one_hot_max_sizes),

        'sampler': GaussianCategoricalSampler(one_hot_max_sizes,
                                              sample_most_probable=True),

        'vlb_scale_factor': 1 / len(one_hot_max_sizes),

        'optimizer': lambda parameters: Adam(parameters, lr=3e-4),

        'mask_generator': MCARGenerator(0.2),

        'proposal_network': proposal_network,

        'prior_network': prior_network,

        'generative_network': generative_network,
    }
