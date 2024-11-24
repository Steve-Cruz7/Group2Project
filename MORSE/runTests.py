# Ok so the plan here is to train and test the MLP they have, but only on the noisy data without their methodology messing with it
# Real World dataset has these specs for get_dataset('./data/real_world', 'malware', 'none', 0,3, 'none', 0.20, 12)
# Synthetic dataset specs for get_dataset('./data/synthetic', 'malware', 'symmetric', 0.6, 'step', 0.20, 12)

import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch 
from dataset import get_dataset
from model import MLP_Net
from utils import AverageMeter
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

def test(testloader, eval_model, epoch, total_epochs, device, num_class, origin):
        eval_model.eval()  # Change model to 'eval' mode.
        correct = 0
        total = 0

        # return the class-level accuracy
        model_preds = []
        model_true = []

        with torch.no_grad():
            for i, (x, y) in enumerate(testloader):
                x = x.to(device)
                y = y.to(device)
                logits = eval_model(x)

                outputs = F.softmax(logits, dim=1)
                _, pred = torch.max(outputs.data, 1)

                total += y.size(0)
                correct += (pred.cpu() == y.cpu().long()).sum()

                # add pred1 | labels
                model_preds.append(pred.cpu())
                model_true.append(y.cpu().long())

        model_preds = np.concatenate(model_preds, axis=0)
        model_true = np.concatenate(model_true, axis=0)

        cm = confusion_matrix(model_true, model_preds)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        class_acc = cm.diagonal()

        acc = 100 * float(correct) / float(total)
        print(class_acc)
        print('Epoch [%3d/%3d] Test Acc: %.2f%%' %(epoch, total_epochs, acc))
        gap_cls = int(num_class / 2)

        if origin != 'real':
           print('Large Class Accuracy is %.2f Small Class Accuracy is %.2f' %(np.mean(class_acc[:gap_cls]), np.mean(class_acc[gap_cls:])))

        return acc, class_acc

def train_and_test(model, epochs, trainloader, testloader, optimizer, criterion, device, num_class, origin):
        for epoch in range(epochs):
            model.train()
            losses = AverageMeter()
            for i, (x, y, _) in enumerate(trainloader):
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                
                optimizer.zero_grad()
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                losses.update(loss.item(), len(logits))

            acc, class_acc = test(testloader, model, epoch, epochs, device, num_class, origin)
            print('Epoch [%3d/%3d] Loss: %.2f' % (epoch, epochs, losses.avg))

        

#dataset = 'real'
#root = './data/real_world'

dataset = 'syn'
root = './data/synthetic'
if dataset == 'real':
    num_classes = 12
    model = MLP_Net(input_dim = 1024, hiddens = [512, 512, num_classes])
    dataset_train, dataset_test, train_data, train_labels, clean_labels = get_dataset(root, dataset = 'malware', noise_type = 'none', noise_rate = 0.3, imb_type = 'none', imb_ratio =  0.2, num_classes = num_classes)
else:
    num_classes = 10
    model = MLP_Net(input_dim = 2381, hiddens = [1024, 1024, num_classes])
    dataset_train, dataset_test, train_data, train_labels, clean_labels = get_dataset(root, dataset = 'malware', noise_type = 'symmetric', noise_rate = 0.6, imb_type = 'step', imb_ratio = 0.2, num_classes= num_classes)


epochs = 100
optimizer = torch.optim.Adam(model.parameters(), 1e-3 , weight_decay= 2e-4) # Learning Rate is in the middle
criterion = nn.CrossEntropyLoss()
device = torch.device('cpu')

train_loader = DataLoader(dataset_train, batch_size = 10, shuffle = True )
test_loader = DataLoader(dataset_test, batch_size = 10, shuffle = True)

train_and_test(model, epochs, train_loader, test_loader, optimizer, criterion, device, num_classes, dataset)





"""for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for i, (batch_X, batch_y, _) in enumerate(train_loader):
        optimizer.zero_grad()
        predictions = model(batch_X)  #Calling the forward method

        loss = criterion(predictions, batch_y)


        loss.backward()   # Derivative Computation

        optimizer.step()
        total_loss += loss.item()

    avg_total_loss = total_loss/ len(train_loader)
    print("Epoch ", epoch, " Avg Loss: ", avg_total_loss)


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        outputs = model(inputs)
        predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Final Test Accuracy: {100 * correct / total:.2f}%')
"""