import torch
import os
from collections import OrderedDict
import numpy as np

def safe_torch_load(weights):
    """
    Safe torch.load wrapper for PyTorch 2.6+ compatibility.
    PyTorch 2.6 changed default weights_only from False to True.
    For trusted checkpoints, we use weights_only=False.
    """
    try:
        # Try with weights_only=False for PyTorch 2.6+
        return torch.load(weights, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions without weights_only parameter
        return torch.load(weights)

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = safe_torch_load(weights)
    # print(checkpoint)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except Exception as e:
        # Try removing 'module.' prefix (from DataParallel)
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # print(k)
            name = k[7:] if k.startswith('module.') else k  # remove `module.` if present
            new_state_dict[name] = v
        
        try:
            model.load_state_dict(new_state_dict)
        except RuntimeError as load_error:
            # Provide helpful error message
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(new_state_dict.keys())
            
            missing_keys = model_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys
            
            print("\n" + "="*80)
            print("ERROR: Checkpoint architecture mismatch!")
            print("="*80)
            print(f"\nCheckpoint file: {weights}")
            
            if missing_keys:
                print(f"\nMissing keys in checkpoint ({len(missing_keys)} keys):")
                for key in list(missing_keys)[:5]:  # Show first 5
                    print(f"  - {key}")
                if len(missing_keys) > 5:
                    print(f"  ... and {len(missing_keys)-5} more")
            
            if unexpected_keys:
                print(f"\nUnexpected keys in checkpoint ({len(unexpected_keys)} keys):")
                for key in list(unexpected_keys)[:5]:  # Show first 5
                    print(f"  - {key}")
                if len(unexpected_keys) > 5:
                    print(f"  ... and {len(unexpected_keys)-5} more")
            
            print("\nPossible solutions:")
            print("  1. Check if you're using the correct model:")
            print("     - For small model: use 'from model_S import MultiscaleNet'")
            print("     - For full model: use 'from model import MultiscaleNet'")
            print("  2. Make sure the checkpoint matches your model architecture")
            print("  3. Use the correct checkpoint file (check training logs)")
            print("="*80 + "\n")
            
            raise load_error


def load_checkpoint_compress_doconv(model, weights):
    checkpoint = safe_torch_load(weights)
    # print(checkpoint)
    # state_dict = OrderedDict()
    # try:
    #     model.load_state_dict(checkpoint["state_dict"])
    #     state_dict = checkpoint["state_dict"]
    # except:
    old_state_dict = checkpoint["state_dict"]
    state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        # print(k)
        name = k
        if k[:7] == 'module.':
            name = k[7:]  # remove `module.`
        state_dict[name] = v
    # state_dict = checkpoint["state_dict"]
    do_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[-1] == 'W' and k[:-1] + 'D' in state_dict:
            k_D = k[:-1] + 'D'
            k_D_diag = k_D + '_diag'
            W = v
            D = state_dict[k_D]
            D_diag = state_dict[k_D_diag]
            D = D + D_diag
            # W = torch.reshape(W, (out_channels, in_channels, D_mul))
            out_channels, in_channels, MN = W.shape
            M = int(np.sqrt(MN))
            DoW_shape = (out_channels, in_channels, M, M)
            DoW = torch.reshape(torch.einsum('ims,ois->oim', D, W), DoW_shape)
            do_state_dict[k] = DoW
        elif k[-1] == 'D' or k[-6:] == 'D_diag':
            continue
        elif k[-1] == 'W':
            out_channels, in_channels, MN = v.shape
            M = int(np.sqrt(MN))
            W_shape = (out_channels, in_channels, M, M)
            do_state_dict[k] = torch.reshape(v, W_shape)
        else:
            do_state_dict[k] = v
    model.load_state_dict(do_state_dict)
def load_checkpoint_hin(model, weights):
    checkpoint = safe_torch_load(weights)
    # print(checkpoint)
    try:
        model.load_state_dict(checkpoint)
    except:
        state_dict = checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
def load_checkpoint_multigpu(model, weights):
    checkpoint = safe_torch_load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = safe_torch_load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = safe_torch_load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    # for p in optimizer.param_groups: lr = p['lr']
    # return lr
