from argparse import ArgumentParser
from importlib import import_module
from os import makedirs
from os.path import join

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from datasets import load_dataset, ZipDatasets
from train_utils import extend_batch_tuple
from VAEAC import VAEAC


parser = ArgumentParser(description='Inpaint images using a given model.')

parser.add_argument('--model_dir', type=str, action='store', required=True,
                    help='Directory with a model and its checkpoints. ' +
                         'It must be a directory in the root ' +
                         'of this repository.')

parser.add_argument('--num_samples', type=int, action='store', default=5,
                    help='Number of different inpaintings per image.')

parser.add_argument('--dataset', type=str, action='store', required=True,
                    help='The name of dataset of images to inpaint ' +
                         '(see load_datasets function in datasets.py)')

parser.add_argument('--masks', type=str, action='store', required=True,
                    help='The name of masks dataset of the same length ' +
                         'as the images dataset. White color (i. e. one ' +
                         'in each channel) means a pixel to inpaint.')

parser.add_argument('--out_dir', type=str, action='store', required=True,
                    help='The name of directory where to save ' +
                         'inpainted images.')

parser.add_argument('--use_last_checkpoint', action='store_true',
                    default=False,
                    help='By default the model with the best ' +
                         'validation IWAE (best_checkpoint.tar) is used ' +
                         'to generate inpaintings. This flag indicates ' +
                         'that the last model (last_checkpoint.tar) ' +
                         'should be used instead.')

args = parser.parse_args()

# Default parameters which are not supposed to be changed from user interface
use_cuda = torch.cuda.is_available()
verbose = True
# Non-zero number of workers cause nasty warnings because of some bug in
# multiprocess library. It might be fixed now, so maybe it is time to set it
# to the number of CPU cores in the system.
num_workers = 0

# import the module with the model networks definitions
model_module = import_module(args.model_dir + '.model')

# build VAEAC on top of the imported networks
model = VAEAC(
    model_module.reconstruction_log_prob,
    model_module.proposal_network,
    model_module.prior_network,
    model_module.generative_network
)
if use_cuda:
    model = model.cuda()
batch_size = model_module.batch_size
sampler = model_module.sampler

# load the required checkpoint
location = 'cuda' if use_cuda else 'cpu'
checkpoint_path = join(args.model_dir,
                       'last_checkpoint.tar' if args.use_last_checkpoint
                       else 'best_checkpoint.tar')
checkpoint = torch.load(checkpoint_path, map_location=location)
model.load_state_dict(checkpoint['model_state_dict'])

# load images and masks datasets, build a dataloader on top of them
dataset = load_dataset(args.dataset)
masks = load_dataset(args.masks)
dataloader = DataLoader(ZipDatasets(dataset, masks), batch_size=batch_size,
                        shuffle=False, drop_last=False,
                        num_workers=num_workers)


# saves inpainting to file
def save_img(img, path):
    ToPILImage()((img / 2 + 0.5).clamp(0, 1).cpu()).save(path)


# create directory for inpaintings, if not exists
makedirs(args.out_dir, exist_ok=True)

iterator = dataloader
if verbose:
    iterator = tqdm(iterator)

image_num = 0
for batch_tuple in iterator:
    batch, masks = batch_tuple
    init_shape = batch.shape[0]

    # if batch size is less than batch_size, extend it with objects
    # from the beginning of the dataset
    batch_tuple_extended = extend_batch_tuple(batch_tuple, dataloader,
                                              batch_size)
    batch_extended, masks_extended = batch_tuple_extended

    if use_cuda:
        batch_extended = batch_extended.cuda()
        masks_extended = masks_extended.cuda()
        batch = batch.cuda()
        masks = masks.cuda()

    # compute imputation distributions parameters
    with torch.no_grad():
        samples_params = model.generate_samples_params(batch_extended,
                                                       masks_extended,
                                                       args.num_samples)
        samples_params = samples_params[:init_shape]

    # save model input, groundtruth and inpaintings to out_dir
    for groundtruth, mask, img_samples_params \
            in zip(batch, masks, samples_params):

        # save groundtruth image
        save_img(groundtruth,
                 join(args.out_dir, '%05d_groundtruth.jpg' % image_num))

        # to show mask on the model input we use gray color
        model_input_visualization = torch.tensor(groundtruth)
        model_input_visualization[mask.byte()] = 0.5

        # save model input visualization
        save_img(model_input_visualization,
                 join(args.out_dir, '%05d_input.jpg' % image_num))

        # in the model input the unobserved part is zeroed
        model_input = torch.tensor(groundtruth)
        model_input[mask.byte()] = 0

        img_samples = sampler(img_samples_params)
        for i, sample in enumerate(img_samples):
            sample[1 - mask.byte()] = 0
            sample += model_input
            sample_filename = join(args.out_dir,
                                   '%05d_sample_%03d.jpg' % (image_num, i))
            save_img(sample, sample_filename)

        image_num += 1
