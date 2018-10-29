import torch
from torch.autograd import Variable
from tqdm import *

def valid(epoch, valid_dataloader, model, crit):
    model.eval()

    running_loss = 0.0
    it = 0
    correct = 0
    total = 0

    for batch in valid_dataloader:
        inputs, targets = batch
        inputs = torch.unsqueeze(inputs, 1)

        inputs = Variable(inputs.float())
        targets = Variable(targets, requires_grad=False)
        with torch.no_grad():
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = model(inputs)
            loss = crit(outputs, targets)

            it += 1
            running_loss += loss.data.item()
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).sum().item()
            total += targets.size(0)

    accuracy = 100 * correct / total
    epoch_loss = running_loss / it

    return epoch_loss, accuracy

def fit(model, train_data, valid_data, crit, optim, num_epochs=1):
    for epoch in range(num_epochs):
        model = model.train()
        pbar = tqdm(train_data, unit="audios", unit_scale=train_data.batch_size)
        
        batch_loss = 0.0
        num_correct = 0
        total = 0
        for i, (batch_x, batch_y) in enumerate(pbar, 1):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            
            model_input = Variable(torch.unsqueeze(batch_x, 1), requires_grad=True).float()
            output = model(model_input)
            
            targets = Variable(batch_y, requires_grad=False)
            loss = crit(output, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            # metrics
            batch_loss += loss.data.item()
            pred = output.data.max(1, keepdim=True)[1]
            num_correct += pred.eq(targets.data.view_as(pred)).sum()
            total += targets.size(0)
            
            pbar.set_postfix({
                'loss': "%.05f" % (batch_loss / i),
                'acc': "%.02f%%" % (100 * num_correct / total)
            })
        epoch_loss, epoch_acc = valid(epoch, valid_data, model, crit)
        print(epoch_loss, epoch_acc)