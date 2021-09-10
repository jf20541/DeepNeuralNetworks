# DeepNeuralNetworks

## Objective
Predict if Hotel Bookings guests would attend or cancel their reservations (binary classification) by using an **Artificial Neural Networks**. Implement the ANN to seek the optimal performance based on the feature selection using Tree-Based Classifier, Fisher Score-ChiSquare Test, and Mutual Information in Classification. Set **ROC-AUC** to evaluate the performance of the model for the imbalanced dataset. **Binary Cross Entropy** as the criterion that measures the error between the target and the output values. Set **Adam Optimizer** as the optimizer for training the model.


The model allows the hospitality industry to find the optimal frontier of services based on attended guests such as food requirements, seasonal patterns, price optimization, availability, employment efficiencies, targeted country marketing, etc.


## Repository File Structure
    ├── src          
    │   ├── train.py                    # Training the NN and evaluating, imports model.py, engine.py, dataset.py, config.py
    │   ├── model.py                    # Neural Networks architecture, inherits nn.Module
    │   ├── engine.py                   # Class Engine for Training, Evaluation, and Loss function 
    │   ├── dataset.py                  # Custom Dataset that return a paris of [input, label] as tensors
    │   └── config.py                   # Define path as global variable
    ├── inputs
    │   ├── hotel_bookings.csv          # Dataset for Hotel Bookings (not cleaned)
    │   ├── hotel_chi2.csv              # Hotel Bookings using Fisher Score-Chi2 feature selection 
    │   ├── hotel_feat_importance.csv   # Hotel Bookings using feature importance (Tree Based) 
    │   ├── hotel_mutual_info.csv       # Hotel Bookings using Information Gain-Mutual Information 
    │   └── train.csv                   # Cleaned Data and Featured Engineered 
    ├── notebooks
    │   └── hotel_booking.ipynb         # Exploratory Data Analysis and Feature Engineering
    ├── models
    │   └── model.bin                   # Neural Networks parameters saved into model.bin 
    ├── requierments.txt                # Packages used for project
    └── README.md
    
## Output
**No Feature Selection**
```
Epoch:15/15, Train ROC-AUC: 0.9021, Eval ROC-AUC: 0.8733
```
**Feature Selection using Tree Based Classifier**
```
Epoch:15/15, Train ROC-AUC: 0.8745, Eval ROC-AUC: 0.7796
```
**Feature Selection using Fisher Score-ChiSquare Test**
```
Epoch:15/15, Train ROC-AUC: 0.8368, Eval ROC-AUC: 0.7521
```
**Feature Selection using Information Gain-Mutual Information in Classification**
```
Epoch:15/15, Train ROC-AUC: 0.8746, Eval ROC-AUC: 0.7843
```

## Model's Architecture
```
NeuralNerwork(
  (fc1): Linear(in_features=30, out_features=15, bias=True)
  (fc2): Linear(in_features=15, out_features=10, bias=True)
  (fc3): Linear(in_features=10, out_features=1, bias=True)
  (dropout): Dropout(p=0.2, inplace=False)
  (sigmoid): Sigmoid()
)
```  

## Terms
- `Epochs:` Number of complete passes of the entire training dataset through the Neural Network
- `Learning Rate:` The step size at each iteration while moving toward a minimum of a loss function
- `Batch Size:` Number of training examples utilized in one forward/backward pass
- `ROC AUC Score:` ROC is a probability curve and AUC represents the degree or measure of separability
- `Adam Optimizer:` A stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments
- `ReLU:` A non-linear activation function that is used in multi-layer neural networks 
- `Sigmoid:` A activation function that maps the entire number line into a small range such as between 0 and 1
- `Linear:` Applies a linear transformation to the incoming data
- `DataLoader:` Iterates through all values and returns in batches.


## Data
  ```
  Binary Targets [0,1]: ['is_canceled']
  Categorical Features:['hotel','arrival_date_month','meal','market_segment','distribution_channel','reserved_room_type','deposit_type','customer_type','country']
  Numerical Features ['lead_time','arrival_date_year','arrival_date_week_number','arrival_date_day_of_month','is_repeated_guest','previous_cancellations','previous_bookings_not_canceled','booking_changes','agent','days_in_waiting_list','adr','required_car_parking_spaces','total_of_special_requests']
  ```
