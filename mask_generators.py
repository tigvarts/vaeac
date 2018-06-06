import numpy as np
import torch

def generate_horizontal_line(batch_size, h, w, line_width):
    mask0 = np.zeros((batch_size, h, w), dtype='float32')
    idx = np.random.randint(0, h - line_width, size=batch_size)
    for i in range(line_width):
        mask0[np.arange(batch_size), idx + i, :] = 1
    mask0 = mask0.reshape(batch_size, -1)
    return 1 - torch.from_numpy(mask0)

def generate_rectangle(batch_size, h, w, reject):
    def gen():
        x1, x2 = np.random.randint(0, h, 2)
        y1, y2 = np.random.randint(0, w, 2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        return int(x1), int(y1), int(x2), int(y2)
    mask0 = torch.zeros(batch_size, 3, h, w)
    for i in range(batch_size):
        x1, y1, x2, y2 = gen()
        while reject(x1, y1, x2, y2):
            x1, y1, x2, y2 = gen()
        mask0[i, :, x1:x2+1, y1:y2+1] = 1
    return mask0.view(batch_size, -1)

def generate_bernoulli(batch_size, d, p=0.9):
    mask = torch.from_numpy(np.random.binomial(1, p, (batch_size, d))).float()
    return mask

def generate_ones(batch_size, d):
    return torch.ones(batch_size, d)

def generate_mixture(batch_size, generators, weights):
    w = np.array(weights, dtype='float')
    w /= w.sum()
    c_ids = np.random.choice(w.size, batch_size, True, w)
    mask = None
    for i, gen in enumerate(generators):
        ids = np.where(c_ids == i)[0]
        if len(ids) == 0:
            continue
        samples = gen(len(ids))
        if mask is None:
            mask = torch.zeros(batch_size, samples.shape[1])
        mask[ids] = samples
    return mask