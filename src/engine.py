import torch
from torch.autograd import Variable

# criterion that measures Binary Cross Entropy between (outputs, targets)
def loss_fn(outputs, targets):
    return torch.nn.BCELoss()(outputs, targets.view(-1, 1))
        
def train_fn(dataloader, model, optimizer):
    # training the model
    model.train()
    for i, (features, targets) in enumerate(dataloader):    
        # wrapping tensor values in Variables        
        features, targets = Variable(features), Variable(targets)
        # compute forward pass through the model
        outputs = model(features)
        # compute loss Binary Cross Entropy
        loss = loss_fn(outputs, targets)   
        # set gradients to 0
        optimizer.zero_grad()
        # compute gradient of loss w.r.t all the parameters
        loss.backward()
        # optimizer iterate over all parameters (updates parameters)
        optimizer.step()

def eval_fn(dataloader, model):
    # Set module in evaluation mode
    model.eval()
    fin_targets = []
    fin_outputs = []
    # disabling tracking of gradients
    with torch.no_grad():
        for i, (features, targets) in enumerate(dataloader):            
            features, targets = Variable(features), Variable(targets)
            outputs = model(features)
            # append to empty list and conver to numpy array  to list 
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets