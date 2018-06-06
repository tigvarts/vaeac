import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

def train_model(train_data, model, generate_mask, generate_weights=None,
                tests=[], test_freq=200, batch_size=128, num_epochs=50,
                learning_rate=1e-3, verbose_update_freq=50, num_workers=0):
    gd = optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    total_batches = len(dataloader)
    train_losses = []
    test_results = []
    model = model.cuda()
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            if isinstance(batch, tuple) or isinstance(batch, list):
                batch = batch[0]
            batch = batch.view(batch.shape[0], -1)
            model.train()
            x, b = Variable(batch), Variable(generate_mask(batch.size(0)))
            if generate_weights:
                w = Variable(generate_weights(x.data, b.data))
            else:
                w = None
            if next(model.parameters()).is_cuda:
                x = x.cuda()
                b = b.cuda()
                if w is not None:
                    w = w.cuda()
            loss = model.batch_loss((x, b), w)
            (-loss).backward()
            train_losses.append(float(loss))
            if i % verbose_update_freq == 0 or i == total_batches - 1:
                print('\rEpoch', epoch, 'Train loss', train_losses[-1],
                      'Batch', i + 1, 'of', total_batches, ' ' * 10, end='', flush=True)
            if i % test_freq == 0:
                cur_test_result = {}
                for test in tests:
                    cur_test_result[test['name']] = test['func'](model)
                test_results.append(cur_test_result)
            gd.step()
            gd.zero_grad()
        print(flush=True)
    model.eval()
    return {
        'model': model.cpu(),
        'train_losses_list': train_losses,
        'test_results': test_results
    }