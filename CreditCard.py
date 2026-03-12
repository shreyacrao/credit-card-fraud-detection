import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

# Load dataset
data = pd.read_csv("creditcard.csv")

# Separate fraud and normal transactions
fraud = data[data.Class == 1]
normal = data[data.Class == 0]

# Handle imbalance by undersampling
normal_sample = resample(normal,
                         replace=False,
                         n_samples=len(fraud),
                         random_state=42)

balanced_data = pd.concat([fraud, normal_sample])

# Shuffle dataset
balanced_data = balanced_data.sample(frac=1, random_state=42)

# Features and labels
X = balanced_data.drop("Class", axis=1)
y = balanced_data["Class"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))