import tqdm
import torch
def train(model, num_epochs, train_dl, valid_dl, track = False, optimizer = None, criterion = None):
    loss_hist_train = [0]*num_epochs
    loss_hist_valid = [0]*num_epochs
    accuracy_hist_train = [0]*num_epochs
    accuracy_hist_valid = [0]*num_epochs
    for epoch in tqdm.tqdm(range(num_epochs)):
        model.train()
        for i, sample_batched in enumerate(train_dl):
            optimizer.zero_grad()
            outputs = model(sample_batched['image'])
            loss = criterion(outputs, sample_batched['label'].unsqueeze(1))
            loss.backward()
            optimizer.step()
            loss_hist_train[epoch] += loss.item()
            accuracy_hist_train[epoch] += (outputs.round() == sample_batched['label'].unsqueeze(1)).sum().item()
        loss_hist_train[epoch] /= len(train_dl)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)
        model.eval()
        with torch.no_grad():
            for i, sample_batched in enumerate(valid_dl):
                outputs = model(sample_batched['image'])
                loss = criterion(outputs, sample_batched['label'].unsqueeze(1))
                loss_hist_valid[epoch] += loss.item()
                accuracy_hist_valid[epoch] += (outputs.round() == sample_batched['label'].unsqueeze(1)).sum().item()
        loss_hist_valid[epoch] /= len(valid_dl)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        # print loss and accuracy for the train set and validation set
        if track:
            print('Epoch [{}/{}], train_Loss: {:.4f}, train_acc: {:.2f}%'.format(epoch+1, num_epochs, loss_hist_train[epoch], accuracy_hist_train[epoch]*100))
            print('val_loss: {:.4f}, val_acc: {:.2f}%'.format(loss_hist_valid[epoch], accuracy_hist_valid[epoch]*100))
    return model, loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid