# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Dharshan D

### Register Number: 212223230045

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
X = torch.linspace(1,50,50).reshape(-1,1)
#X
torch.manual_seed(71)
e = torch.randint(-8,9,(50,1),dtype=torch.float)
#e
y = 2*X + 1 + e
plt.scatter(X.numpy(), y.numpy(),color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regression')
plt.show()

# Define the Linear Model Class
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    # Initialize the Model
torch.manual_seed(59)  # Ensure same initial weights
model = Model(1, 1)
print('Weight:', model.linear.weight.item())
print('Bias:  ', model.linear.bias.item())
initial_weight = model.linear.weight.item()
initial_bias = model.linear.bias.item()
print("\nName:Dharshan D ")
print("Register No: 212223230045")
print(f'Initial Weight: {initial_weight:.8f}, Initial Bias: {initial_bias:.8f}\n')
# Define Loss Function & Optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
# Train the Model
epochs = 50
losses = []

for epoch in range(1, epochs + 1):  # Loop over epochs
    y_pred = model.forward(X)
    loss = loss_function(y_pred,y)
    losses.append(loss)
    # Print loss, weight, and bias for EVERY epoch
    print(f'epoch: {epoch:2}  loss: {loss.item():10.8f}  '
          f'weight: {model.linear.weight.item():10.8f}  '
          f'bias: {model.linear.bias.item():10.8f}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
# Plot Loss Curve

plt.plot(range(epochs), [l.item() for l in losses], color='blue')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss Curve')
plt.show()
# Final Weights & Bias
final_weight = model.linear.weight.item()
final_bias = model.linear.bias.item()
print("\nName:Dharshan D ")
print("Register No: 212223230045")
print(f'\nFinal Weight: {final_weight:.8f}, Final Bias: {final_bias:.8f}')
#  Best-Fit Line Calculation
x1 = torch.tensor([X.min().item(), X.max().item()]) # Find min and max values of X
y1 = x1 * final_weight + final_bias # Compute corresponding y-values using trained model
# Plot Original Data & Best-Fit Line
plt.scatter(X, y, label="Original Data")
plt.plot(x1, y1, 'r', label="Best-Fit Line")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trained Model: Best-Fit Line')
plt.legend()
plt.show()
# Prediction for x = 120
x_new = torch.tensor([[120.0]])  # New input as a tensor
y_new_pred = model(x_new).item()  # Predict using trained model
print("\nName:Dharshan D ")
print("Register No: 212223230045")
print(f"\nPrediction for x = 120: {y_new_pred:.8f}")

```

### Dataset Information

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/f601fa65-88b5-4b81-a17e-c48f277eeb40" />


### OUTPUT
T

### OUTPUT

Dataset Information
<br/>

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/7a9ee54c-310f-46cf-9aba-33443319270e" />

Training Loss Vs Iteration Plot
<br/>

<img width="571" height="455" alt="image-1" src="https://github.com/user-attachments/assets/d61bb30d-beb1-4351-9c0d-d3bb608f1d25" />

Loss Curve
<br/>

<img width="580" height="455" alt="484379878-6e501cc5-9007-4283-a14b-747e30bd223e" src="https://github.com/user-attachments/assets/96126463-6b61-428e-a048-cddd0cf1b59e" />



Best Fit line plot
<br/>



<img width="405" height="108" alt="name" src="https://github.com/user-attachments/assets/3b536f5a-1c9e-4cec-9972-74f931dc7efb" />



## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
