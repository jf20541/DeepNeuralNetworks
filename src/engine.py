import torch
from torch.autograd import Variable


def loss_fn(outputs, targets):
    return torch.nn.BCELoss()(outputs, targets.view(-1, 1))


def train_fn(dataloader, model, optimizer):
    model.train()
    for i, (features, targets) in enumerate(dataloader):
        features, targets = Variable(features), Variable(targets)
        outputs = model(features)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_fn(dataloader, model):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for i, (features, targets) in enumerate(dataloader):
            features, targets = Variable(features), Variable(targets)
            outputs = model(features)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets
