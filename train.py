import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models import get_resnet, count_parameters
from data import get_loaders
from utils import load_checkpoint, set_seed, save_checkpoint, load_config

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        student_soft = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = torch.nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        distill_loss = torch.nn.functional.kl_div(student_soft, teacher_soft, reduction='batchmean')
        distill_loss *= (self.temperature ** 2)

        ce_loss = self.ce(student_logits, labels)

        total_loss = self.alpha * distill_loss + (1 - self.alpha) * ce_loss
        return total_loss, distill_loss, ce_loss

def train_epoch(model, teacher, train_loader, criterion, optimizer, scaler, device, use_amp, is_distillation):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                student_outputs = model(inputs)
                if is_distillation:
                    with torch.no_grad():
                        teacher_outputs = teacher(inputs)
                    loss, _, _ = criterion(student_outputs, teacher_outputs, labels)
                else:
                    loss = criterion(student_outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            student_outputs = model(inputs)
            if is_distillation:
                with torch.no_grad():
                    teacher_outputs = teacher(inputs)
                loss, _, _ = criterion(student_outputs, teacher_outputs, labels)
            else:
                loss = criterion(student_outputs, labels)
            
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return running_loss / len(train_loader)

def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100.0 * correct / total

def train(model, teacher, train_loader, val_loader, config, exp_id, is_distillation, resume_from=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if teacher:
        teacher = teacher.to(device)
        teacher.eval()
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    if is_distillation:
        criterion = DistillationLoss(config['temperature'], config['alpha'])
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Mixed precision
    use_amp = config.get('mixed_precision', False) and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None

    start_epoch = 1
    best_acc = 0.0
    
    if resume_from and os.path.exists(resume_from):
        start_epoch, best_acc = load_checkpoint(resume_from, model, optimizer, scheduler)
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        # Check for existing checkpoint in default location
        default_checkpoint = f"checkpoints/{exp_id}/best.pth"
        if os.path.exists(default_checkpoint):
            start_epoch, best_acc = load_checkpoint(default_checkpoint, model, optimizer, scheduler)
            print(f"Found existing checkpoint for {exp_id}, resuming from epoch {start_epoch + 1}")
    
    # Training loop
    for epoch in range(start_epoch + 1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        
        # Train
        train_loss = train_epoch(
            model, teacher, train_loader, criterion, optimizer, 
            scaler, device, use_amp, is_distillation
        )
        
        # Validate
        val_acc = validate(model, val_loader, device)
        
        # Scheduler step
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            if config.get('save_checkpoints', True):
                checkpoint_dir = f"checkpoints/{exp_id}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_checkpoint(model, optimizer, epoch, val_acc, f"{checkpoint_dir}/best.pth")
                print(f"Saved best model (accuracy: {best_acc:.2f}%)")
        
        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_dir = f"checkpoints/{exp_id}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_checkpoint(model, optimizer, epoch, val_acc, f"{checkpoint_dir}/epoch_{epoch}.pth")
            print(f"Saved checkpoint at epoch {epoch}")
    
    return best_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['baseline', 'distill'],
                        help='Training mode: baseline or distillation')
    parser.add_argument('--teacher_depth', type=int, help='Teacher ResNet depth (distillation only)')
    parser.add_argument('--student_depth', type=int, required=True, help='Student ResNet depth')
    parser.add_argument('--exp_id', type=str, required=True, help='Experiment ID for saving')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--teacher_checkpoint', type=str, help='Path to pretrained teacher (distillation only)')
    parser.add_argument('--resume', type=str, help='Path to student checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    set_seed(config['seed'])
    
    # Create models
    student = get_resnet(args.student_depth, config['num_classes'])
    teacher = None
    
    if args.mode == 'distill':
        if not args.teacher_depth:
            raise ValueError("--teacher_depth required for distillation mode")
        teacher = get_resnet(args.teacher_depth, config['num_classes'])
        
        # Load teacher checkpoint if provided
        if args.teacher_checkpoint:
            checkpoint = torch.load(args.teacher_checkpoint, map_location='cpu')
            teacher.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded teacher from {args.teacher_checkpoint}")
        else:
            print("Warning: No teacher checkpoint provided. Using untrained teacher.")
    
    # Load data
    train_loader, val_loader = get_loaders(
        batch_size=config['batch_size'],
        num_workers=2
    )
    
    # Print model info
    print(f"\nMode: {args.mode.upper()}")
    print(f"Student: ResNet{args.student_depth} ({count_parameters(student):,} params)")
    if teacher:
        print(f"Teacher: ResNet{args.teacher_depth} ({count_parameters(teacher):,} params)")
        print(f"Capacity Ratio: {count_parameters(teacher) / count_parameters(student):.2f}")

    best_acc = train(
        model=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        exp_id=args.exp_id,
        is_distillation=(args.mode == 'distill'),
        resume_from=args.resume  # Pass resume path
    )
    
    print(f"\n✓ Experiment {args.exp_id} completed!")
    print(f"Best accuracy: {best_acc:.2f}%")
    
    # Save results
    results = {
        'exp_id': args.exp_id,
        'mode': args.mode,
        'student_depth': args.student_depth,
        'teacher_depth': args.teacher_depth if teacher else None,
        'best_accuracy': best_acc,
        'ratio': count_parameters(teacher) / count_parameters(student) if teacher else None
    }
    
    # Save to file
    os.makedirs('results', exist_ok=True)
    with open(f"results/{args.exp_id}.yaml", 'w') as f:
        yaml.dump(results, f)

if __name__ == '__main__':
    main()
