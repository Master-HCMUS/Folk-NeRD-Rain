import os
import argparse
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils
from data_RGB import get_test_data
from model import MultiscaleNet as FullNet
from model_S import MultiscaleNet as SmallNet
from skimage import img_as_ubyte
from get_parameter_number import get_parameter_number
from tqdm import tqdm
from layers import *

parser = argparse.ArgumentParser(description='Image Deraining')
parser.add_argument('--input_dir', default='./Datasets/Rain200L/test/input/', type=str, help='Directory of validation images')
parser.add_argument('--output_dir', default='./results/Rain200L', type=str, help='Directory of validation images')
parser.add_argument('--weights', default='', type=str, help='Path to checkpoint file') 
parser.add_argument('--model', default='full', type=str, choices=['full', 'small'], help='Model type: full or small')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--win_size', default=256, type=int, help='window size')
args = parser.parse_args()
result_dir = args.output_dir
win = args.win_size
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# Select model based on argument
if args.model == 'small':
    print("Using Small Model (model_S.py)")
    model_restoration = SmallNet()
else:
    print("Using Full Model (model.py)")
    model_restoration = FullNet()

get_parameter_number(model_restoration)

# Load checkpoint and check for EMA weights
# Handle PyTorch 2.6+ weights_only parameter change
try:
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
except TypeError:
    # Fallback for older PyTorch versions
    checkpoint = torch.load(args.weights, map_location='cpu')

if isinstance(checkpoint, dict):
    if 'ema_shadow' in checkpoint:
        print("===>Found EMA shadow weights, loading them for better results")
        # Load EMA shadow weights
        ema_weights = checkpoint['ema_shadow']
        model_state = model_restoration.state_dict()
        for name in model_state.keys():
            if name in ema_weights:
                model_state[name] = ema_weights[name]
        model_restoration.load_state_dict(model_state)
    elif 'state_dict' in checkpoint:
        print("===>Loading regular checkpoint weights")
        utils.load_checkpoint(model_restoration, args.weights)
    else:
        print("===>Loading checkpoint (unknown format)")
        model_restoration.load_state_dict(checkpoint)
else:
    print("===>Loading checkpoint directly")
    model_restoration.load_state_dict(checkpoint)

print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# dataset = args.dataset
rgb_dir_test = args.input_dir
test_dataset = get_test_data(rgb_dir_test, img_options={})
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)


utils.mkdir(result_dir)

with torch.no_grad():
    psnr_list = []
    ssim_list = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        input_    = data_test[0].cuda()
        filenames = data_test[1]
        _, _, Hx, Wx = input_.shape
        filenames = data_test[1]
        input_re, batch_list = window_partitionx(input_, win)
        restored = model_restoration(input_re)
        restored = window_reversex(restored[0], win, Hx, Wx, batch_list)

        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        for batch in range(len(restored)):
            restored_img = restored[batch]
            restored_img = img_as_ubyte(restored[batch])
            utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)
