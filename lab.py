from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd

# Load dataset
df = pd.read_csv("C:\Users\Venu Mirapakayala\Desktop\lab\lab\Fish.csv")

# Display first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Encode species column (convert categorical column to numerical labels)
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

# Define Features (X) and Target Variable (y)
X = df.drop(columns=["Species"])  # Features
y = df["Species"]  # Target (Classification target)

# Split Data for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train classification model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate performance
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(model, "fish_classification_model.pkl")
print("Classification model saved successfully!")

# Save label encoder to map numerical species back to names
joblib.dump(le, "species_label_encoder.pkl")
print("Label encoder saved!")
