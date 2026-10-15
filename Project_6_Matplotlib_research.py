
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, plot_tree

# Sample data for a basic AI model
data = {
    "Age": [25, 35, 45, 55, 65, 75],
    "BMI": [22, 27, 30, 32, 29, 35],
    "Blood_Sugar": [90, 110, 130, 145, 160, 180],
    "At_Risk": [0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Separate input features and target
X = df[["Age", "BMI", "Blood_Sugar"]]
y = df["At_Risk"]

# Train the decision tree
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# Create a dashboard window
plt.figure(figsize=(14, 6))

# Plot 1: Decision Tree
plt.subplot(1, 2, 1)

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Healthy", "At Risk"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree Rules")

# Plot 2: Feature Importance
plt.subplot(1, 2, 2)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")

# Adjust spacing and display the pop-up
plt.tight_layout()
plt.show()
