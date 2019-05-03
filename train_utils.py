import torch
from tqdm import tqdm


def extend_batch(batch, dataloader, batch_size):
    """
    If the batch size is less than batch_size, extends it with
    data from the dataloader until it reaches the required size.
    Here batch is a tensor.
    Returns the extended batch.
    """
    while batch.shape[0] != batch_size:
        dataloader_iterator = iter(dataloader)
        nw_batch = next(dataloader_iterator)
        if nw_batch.shape[0] + batch.shape[0] > batch_size:
            nw_batch = nw_batch[:batch_size - batch.shape[0]]
        batch = torch.cat([batch, nw_batch], 0)
    return batch


def extend_batch_tuple(batch, dataloader, batch_size):
    """
    The same as extend_batch, but here the batch is a list of tensors
    to be extended. All tensors are assumed to have the same first dimension.
    Returns the extended batch (i. e. list of extended tensors).
    """
    while batch[0].shape[0] != batch_size:
        dataloader_iterator = iter(dataloader)
        nw_batch = next(dataloader_iterator)
        if nw_batch[0].shape[0] + batch[0].shape[0] > batch_size:
            nw_batch = [nw_t[:batch_size - batch[0].shape[0]]
                        for nw_t in nw_batch]
        batch = [torch.cat([t, nw_t], 0) for t, nw_t in zip(batch, nw_batch)]
    return batch


def get_validation_iwae(val_dataloader, mask_generator, batch_size,
                        model, num_samples, verbose=False):
    """
    Compute mean IWAE log likelihood estimation of the validation set.
    Takes validation dataloader, mask generator, batch size, model (VAEAC)
    and number of IWAE latent samples per object.
    Returns one float - the estimation.
    """
    cum_size = 0
    avg_iwae = 0
    iterator = val_dataloader
    if verbose:
        iterator = tqdm(iterator)
    for batch in iterator:
        init_size = batch.shape[0]
        batch = extend_batch(batch, val_dataloader, batch_size)
        mask = mask_generator(batch)
        if next(model.parameters()).is_cuda:
            batch = batch.cuda()
            mask = mask.cuda()
        with torch.no_grad():
            iwae = model.batch_iwae(batch, mask, num_samples)[:init_size]
            avg_iwae = (avg_iwae * (cum_size / (cum_size + iwae.shape[0])) +
                        iwae.sum() / (cum_size + iwae.shape[0]))
            cum_size += iwae.shape[0]
        if verbose:
            iterator.set_description('Validation IWAE: %g' % avg_iwae)
    return float(avg_iwae)
