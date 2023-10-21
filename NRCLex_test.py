from sentiment import tokenizeNRCLex
from APP_model import *
def train(model, optimiser, train_loader, epoch, device):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        target = tokenizeNRCLex(data).to(device)
        # print(data.shape)
        # print(target.shape)
        optimiser.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimiser.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                   epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))