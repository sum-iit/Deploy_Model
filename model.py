# model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_and_save_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, 'iris_model.pkl')
    print("Model saved as 'iris_model.pkl'")

if __name__ == "__main__":
    train_and_save_model()
