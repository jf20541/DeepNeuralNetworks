# DeepNeuralNetworks

## Objective
Use Artificial Neural Network for a Binary Classification problem by predicting Hotel Bookings Data guests would attend or cancel their reservation. The evaluation metric for the model is **ROC-AUC** since the target values are skewed. The criterion that measures the error between the target and the output values is **Binary Cross Entropy**. The optimizer is **Adam** which is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.\

The model allows the hospitality industry to find the optimal frontier of services based on attended guests such as food requirements, seasonal patterns, price optimization, availability, employment efficiencies, targeted country marketing, etc.


## Repository File Structure
    ├── src          
    │   ├── train.py             # Training the NN and evaluating, imports model.py, engine.py, dataset.py, config.py
    │   ├── model.py             # Neural Networks architecture, inherits nn.Module
    │   ├── engine.py            # Class Engine for Training, Evaluation, and Loss function 
    │   ├── dataset.py           # Custom Dataset that return a paris of [input, label] as tensors
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
    
## Output
```bash
       ....                   ....                  ....
Epoch:10/15, Train ROC-AUC: 0.8904, Eval ROC-AUC: 0.8565
Epoch:11/15, Train ROC-AUC: 0.8939, Eval ROC-AUC: 0.8615
Epoch:12/15, Train ROC-AUC: 0.8967, Eval ROC-AUC: 0.8684
Epoch:13/15, Train ROC-AUC: 0.8986, Eval ROC-AUC: 0.8680
Epoch:14/15, Train ROC-AUC: 0.9007, Eval ROC-AUC: 0.8708
Epoch:15/15, Train ROC-AUC: 0.9021, Eval ROC-AUC: 0.8733
```

## Model
```
NeuralNerwork(
  (fc1): Linear(in_features=30, out_features=15, bias=True)
  (fc2): Linear(in_features=15, out_features=10, bias=True)
  (fc3): Linear(in_features=10, out_features=1, bias=True)
  (dropout): Dropout(p=0.2, inplace=False)
  (sigmoid): Sigmoid()
)
```  

## Parameters
- `Epochs:` 
- `Learning Rate:`
- `Batch Size:`
