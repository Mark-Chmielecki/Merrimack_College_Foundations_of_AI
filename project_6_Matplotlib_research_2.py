import matplotlib.pyplot as plt

# Create the figure
plt.figure(figsize=(12, 12))

# Plot 1: Top left
plt.subplot(3, 2, 1)
plt.title("Feature Importance - Decision Tree Classifier Model")

# Plot 2: Top right
plt.subplot(3, 2, 2)
plt.title("Decision Tree Rules - Decision Tree  Classifier Model")

# Plot 3: Middle left
plt.subplot(3, 2, 3)
plt.title("Feature Importance - Random Forest Classifier Model")

# Plot 4: Middle right
plt.subplot(3, 2, 4)
plt.title("Decision Tree Rules - Random Forest Classifier Model")

# Plot 5: Bottom left
plt.subplot(3, 2, 5)
plt.title("Feature Importance - K-Nearest Neighbors Classifier Model")

# Plot 6: Bottom right
plt.subplot(3, 2, 6)
plt.title("Decision Tree Rules - K-Nearest Neighbors Classifier Model")

# Adjust spacing and display all plots
#plt.tight_layout()
plt.subplots_adjust(hspace=0.5,wspace=0.3) #horizontal and vertical spacing between subplots

plt.show()