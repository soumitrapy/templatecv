import torch
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

def train_one_epoch(model, dl, optimizer, loss_fn, config, epoch=1, device='cpu'):
    model.to(device)
    model.train()
    running_loss = 0.0
    correct, total  = 0,0
    pbar = tqdm(dl,desc=f"Epoch {epoch+1}")
    for i, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
        avg_loss = running_loss/(i+1)
        acc = 100.*correct/total

        if config['log_interval']>0 and i % config['log_interval'] == 0:
            pbar.set_postfix(train_loss=avg_loss, train_accuracy=acc)
    
    return avg_loss, acc

def val_one_epoch(model, dl, loss_fn, config, epoch=1, device='cpu'):
    model.to(device)
    model.eval()
    running_loss = 0.0
    correct, total  = 0,0
    pbar = tqdm(dl,desc=f"Validation: ")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

            avg_loss = running_loss/(i+1)
            acc = 100.*correct/total

            if config['log_interval']>0 and i % config['log_interval'] == 0:
                pbar.set_postfix(val_loss=avg_loss, val_accuracy=acc)
    
    return avg_loss, acc



def train(model, traindl, optimizer, loss_fn, config, scheduler=None, valdl=None, device='cpu'):
    model.to(device)
    best_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)
    for epoch in range(config['epochs']):
        model.train()
        train_loss, train_acc = train_one_epoch(model, traindl, optimizer, loss_fn, config, epoch=epoch, device=device)
        if valdl and (epoch+1)%config['val_interval']==0:
            val_loss, val_acc = val_one_epoch(model, valdl, loss_fn, config, epoch=epoch, device=device)
            if val_loss<best_loss:
                best_loss = val_loss
                model_name = type(model).__name__+'_'+device.type+str(datetime.now())[:15]
                model_path = os.path.join('checkpoints', model_name)
                torch.save(model.state_dict(), model_path)
                
        if scheduler:
            scheduler.step()

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model,dl, class_names, num_images=6, device='cpu'):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dl):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)