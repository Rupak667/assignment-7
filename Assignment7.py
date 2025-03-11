import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Step 0: Read the data into a pandas dataframe
df = pd.read_csv('suv.csv')

# Step 1: Pick Age and Estimated Salary as the features and Purchased as the target variable
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Step 2: Split the data into training and testing sets with 80/20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Scale the features using standard scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train a decision tree classifier with entropy criterion and predict on test set
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X_train, y_train)
y_pred_entropy = dt_entropy.predict(X_test)

# Step 5: Print the confusion matrix and the classification report
conf_matrix_entropy = confusion_matrix(y_test, y_pred_entropy)
class_report_entropy = classification_report(y_test, y_pred_entropy)

print("Confusion Matrix (Entropy Criterion):\n", conf_matrix_entropy)
print("Classification Report (Entropy Criterion):\n", class_report_entropy)

# Step 6: Repeat steps 4 and 5 with the gini criterion
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)

conf_matrix_gini = confusion_matrix(y_test, y_pred_gini)
class_report_gini = classification_report(y_test, y_pred_gini)

print("Confusion Matrix (Gini Criterion):\n", conf_matrix_gini)
print("Classification Report (Gini Criterion):\n", class_report_gini)

# Step 7: Discuss the performance of your models
print("Performance Comparison:")
print("Entropy Criterion Model:")
print("Confusion Matrix:\n", conf_matrix_entropy)
print("Classification Report:\n", class_report_entropy)

print("\nGini Criterion Model:")
print("Confusion Matrix:\n", conf_matrix_gini)
print("Classification Report:\n", class_report_gini)