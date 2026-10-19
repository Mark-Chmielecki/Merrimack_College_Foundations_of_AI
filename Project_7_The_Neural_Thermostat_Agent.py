"""
Created by Mark Chmielecki for CS6313 - Foundations of Artificial Intelligence
Project #7 due Monday, October 19, 2026.   

Project requirements:

1) Build the brain of a Smart Home Thermostat.  The thermostat decides whether to turn the Air Conditioning ON (1) or OFF (0) based on two environmental factors:
-Indoor Temperature (scaled 0-1)
-Number of People in the room (scaled 0-1)

2) Implement a sigmoid activation function: σ(z) = 1 / (1 + e^-z)

3) Predict the probability of Air Conditioning ON by implementing a forward pass based on the weighted sum z = x1*w1 + x2*w2 + bias, passed through the sigmoid.

4) Train the models on 2000 epochs using gradient descent to adjust the weights and bias based on prediction error.

5) Plot using Matplotlib to visualize the loss curve
"""

import numpy as np                          # NumPy library required for the training arrays and the exponent in the Sigmoid formula

import matplotlib.pyplot as plt             # Required for the visualization exercise displayed at the end of the routine

 

class NeuralThermostat:

    # Using object-oriented programming to organize the thermostat's behavior into a class.

    def __init__(self):

        # Sets initial guesses for the NeuralThermostat class, to be adjusted during training

        self.w1 = 0.5                   # initial weight assigned to the indoor temperature
        self.w2 = -0.2                  # initial weignt assigned to the number of people in the 
        self.bias = 0.0                 # initial bias
        self.learning_rate = 0.5        # controls how much the model changes it parameters during training
      

    def sigmoid(self, z):

        # TASK 1: Implement the Sigmoid Activation Function
        # Formula: 1 / (1 + e^-z)
        # Hint: Use np.exp(-z) for the exponential part

        # The sigmoid is an activation function used to transform a weighted sum into a value between zero and one.
        # For example, a sigmoid output of 0.90 can be interpreted as a high model score for AC ON.

        sigmoidActivation =  1 / (1 + np.exp(-z))

        return sigmoidActivation
    

    def predict(self, x1, x2):

        # TASK 2: Implement the Feedforward Pass

        # The predict method is responsible for making a prediction based on the current model parameters.

        # 1. Calculate Z using the formula: (x1 * w1) + (x2 * w2) + b
        z = (x1 * self.w1) + (x2 * self.w2) + self.bias

        # 2. Pass Z through your sigmoid function and return the result
        # The sigmoid function converts z into the final output.
        return self.sigmoid(z)

    def train(self, X, y, epochs=2000):

        # The train method teaches the neural network by comparing its predictions with known correct answers.
        # The model trains for 2,000 epochs.
        # Losses stores the average squared error for each epoch so I can plot the learning curve.
        # total_loss accumulates the errors for the current epoch.
        # Training repeatedly exposes the model to examples so it can adjust its parameters to improve its predictions.

        losses = []

        for epoch in range(epochs):

            total_loss = 0

            for i in range(len(X)):

                x1, x2 = X[i]       # inputs

                target = y[i]       # correct answer based on iputs

                # --- TASK 3: THE TRAINING ENGINE ---

                # 1. Get the current prediction (Call your predict function)
                # The model uses the current weights and bias to make a prediction.

                prob = self.predict(x1, x2)

                # 2. Calculate the Error (The difference between prediction and target)
                # The difference between prediction and target provides a signal for learning.

                error = prob - target

                # 3. Calculate the sigmoid derivative and gradient
                # Multiply prediction error by derivative to calculate a gradient-like adjustment signal.
                # Signal is used when updating weights and bias.

                sigmoid_derivative = prob * (1 - prob)

                gradient = error * sigmoid_derivative


                # 4. Update the weights and bias (Gradient Descent)
                # Formula: Weight = Weight - (Learning_Rate * Gradient * Input)
                # This is where the model learns from its error.

                self.w1 = self.w1 - (self.learning_rate * gradient * x1)

                self.w2 = self.w2 - (self.learning_rate * gradient * x2)    

                self.bias = self.bias - (self.learning_rate * gradient)


                # 5. Record the Squared Error for the loss chart
                # Code squares the error so negative and positive errors both contribute positively to the loss.

                total_loss += (error ** 2)


            # Keep track of the average loss for this epoch
            # At end of each epoch, code divides the accumulated squared error by the number of training examples.

            losses.append(total_loss / len(X))

        return losses

# --- PROJECT DATASET ---

# Features: [Temperature (0-1), PeopleCount (0-1)]

# Targets: 1 (AC On), 0 (AC Off)

X_train = np.array([

    [0.1, 0.1], # Cold & Empty -> Off (0)

    [0.9, 0.2], # Hot & Few people -> On (1)

    [0.5, 0.8], # Warm & Crowded -> On (1)

    [0.2, 0.9]  # Cold & Crowded -> Off (0)

])

y_train = np.array([0, 1, 1, 0])

 

# --- EXECUTION ---

agent = NeuralThermostat()

 

print("--- Before Training ---")

print(f"Initial Weights: w1={agent.w1:.2f}, w2={agent.w2:.2f}, b={agent.bias:.2f}")

# Test prediction for [Hot, Few People]

initial_pred = agent.predict(0.9, 0.2)

print(f"Prediction for [0.9, 0.2]: {initial_pred if initial_pred is not None else 'No Output Yet'}")

 

# Train the agent

print("\nTraining in progress...")

history = agent.train(X_train, y_train)

 

print("\n--- After Training ---")

print(f"Final Weights: w1={agent.w1:.2f}, w2={agent.w2:.2f}, b={agent.bias:.2f}")

final_pred = agent.predict(0.9, 0.2)

print(f"Prediction for [0.9, 0.2]: {final_pred if final_pred is not None else 'No Output Yet'}")

 

# --- VISUALIZATION ---

if history:

    plt.figure(figsize=(10, 5))

    plt.plot(history, color='blue', linewidth=2)

    plt.title("Neural Agent Learning Curve", fontsize=14)

    plt.xlabel("Epochs", fontsize=12)

    plt.ylabel("Mean Squared Error (Loss)", fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()





