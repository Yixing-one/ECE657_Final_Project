import os
import random
import yaml
import torch
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_checkpoint(model, optimizer, epoch, accuracy, path, scheduler=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Loaded checkpoint from {path} (epoch {start_epoch}, accuracy {checkpoint['accuracy']:.2f}%)")
    return start_epoch, checkpoint['accuracy']

def get_latest_checkpoint(exp_id, checkpoint_dir='checkpoints'):
    exp_path = os.path.join(checkpoint_dir, exp_id)
    if not os.path.exists(exp_path):
        return None
    
    checkpoint_files = [f for f in os.listdir(exp_path) if f.endswith('.pth')]
    if not checkpoint_files:
        return None
    
    # Prioritize best.pth, then latest epoch
    if 'best.pth' in checkpoint_files:
        return os.path.join(exp_path, 'best.pth')
    
    # Otherwise get the one with highest epoch number
    epoch_files = [f for f in checkpoint_files if f.startswith('epoch_')]
    if epoch_files:
        epoch_nums = [int(f.split('_')[1].split('.')[0]) for f in epoch_files]
        latest_idx = np.argmax(epoch_nums)
        return os.path.join(exp_path, epoch_files[latest_idx])
    
    return None

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
