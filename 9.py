from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the Iris dataset
dataset = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset["data"], dataset["target"], random_state=0)

# Create a K-Neighbors Classifier with 3 neighbors
kn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
kn.fit(X_train, y_train)

# Make predictions on the testing data
prediction = kn.predict(X_test)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, prediction)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

accuracy = accuracy_score(y_test, prediction)
print("Accuracy:", accuracy)




#output

array([[13, 0, 0],
 [ 0, 15, 1],
 [ 0, 0, 9]], dtype=int64)
