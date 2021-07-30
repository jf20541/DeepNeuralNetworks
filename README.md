# DeepNeuralNetworks

## Objective


## Repository File Structure
    ├── src          
    │   ├── train.py             # Training module, connects model, engine, dataset, and config
    │   ├── model.py             # Neural Networks architecture, inherits nn.Module
    │   ├── engine.py            # For loop for Training, Evaluation, and Loss function 
    │   ├── dataset.py           # Custom Dataset that inherits Torch Dataset to create a custom dataloader
    │   └── config.py            # Define path as global variable
    ├── inputs
    │   ├── hotel_bookings.csv   # Dataset for Hotel Bookings (not cleaned)
    │   └── train.csv            # Cleaned Data and Featured Engineered 
    ├── notebooks
    │   └── hotel_booking.ipynb  # Exploratory Data Analysis and Feature Engineering
    ├── models
    │   └── model.bin            # Neural Networks parameters saved into model.bin 
    ├── requierments.txt         # Packages used for project
    └── README.md

## Model
```
NeuralNerwork(
  (fc1): Linear(in_features=30, out_features=10, bias=True)
  (fc2): Linear(in_features=10, out_features=8, bias=True)
  (fc3): Linear(in_features=8, out_features=4, bias=True)
  (fc4): Linear(in_features=4, out_features=1, bias=True)
  (sigmoid): Sigmoid())
```  

## Metric & Mathematics


## Output
```bash
Epoch:10/15, ROC AUC:0.8565
Epoch:11/15, ROC AUC:0.8615
Epoch:12/15, ROC AUC:0.8684
Epoch:13/15, ROC AUC:0.8680
Epoch:14/15, ROC AUC:0.8708
Epoch:15/15, ROC AUC:0.8733
```


## Parameters
- `Epochs:` 
- `Learning Rate:`
- `Batch Size:`
