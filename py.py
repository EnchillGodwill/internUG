import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from flask import Flask, request, jsonify
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Simulate Data
def simulate_data(num_logs=10000, anomaly_percentage=0.05):
    normal_mean = [50, 30, 5]
    normal_cov = [[10, 4, 1], [4, 10, 2], [1, 2, 1]]
    anomaly_mean = [90, 5, 50]
    anomaly_cov = [[20, 2, 5], [2, 20, 1], [5, 1, 10]]

    normal_logs = np.random.multivariate_normal(normal_mean, normal_cov, int(num_logs * (1 - anomaly_percentage)))
    anomaly_logs = np.random.multivariate_normal(anomaly_mean, anomaly_cov, int(num_logs * anomaly_percentage))

    logs = np.vstack((normal_logs, anomaly_logs))
    labels = np.hstack((np.zeros(normal_logs.shape[0]), np.ones(anomaly_logs.shape[0])))

    indices = np.arange(logs.shape[0])
    np.random.shuffle(indices)
    logs = logs[indices]
    labels = labels[indices]

    df = pd.DataFrame(logs, columns=['login_attempts', 'data_transferred_MB', 'failed_logins'])
    df['label'] = labels
    return df

# Preprocess Data
def preprocess_data(df):
    X = df.drop(columns=['label'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

# Train Model
def train_model(X_train, contamination):
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    model.fit(X_train)
    return model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred == -1, 1, 0)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save Model
def save_model(model, scaler, filename='anomaly_detection_model.pkl'):
    joblib.dump({'model': model, 'scaler': scaler}, filename)

# Flask API
app = Flask(__name__)
model_data = joblib.load('anomaly_detection_model.pkl')
model = model_data['model']
scaler = model_data['scaler']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)
    return jsonify({'anomaly': int(prediction[0] == -1)})

if __name__ == '__main__':
    df = simulate_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    model = train_model(X_train, contamination=0.05)
    evaluate_model(model, X_test, y_test)
    save_model(model, scaler)
    app.run(debug=True)
