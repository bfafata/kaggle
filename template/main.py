import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


"""
def main():
    # Load data
    train_path = "train.csv"
    test_path = "test.csv"
    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)

    # Preprocess data
    train_inputs, train_labels, _ = preprocess(train_frame)
    test_inputs, ids = preprocess(test_frame)

    # Split the training data into training set and validation set
    X_train, X_val, y_train, y_val = train_test_split(train_inputs, train_labels, test_size=0.2, random_state=42)

    # Train the model
    .fit(X_train, y_train)

    # Predict on validation data and print metrics
    y_val_pred = clf.predict(X_val)
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred)}")
    print(classification_report(y_val, y_val_pred, target_names=["dead", "alive"]))

    # Predict on test data
    y_pred = clf.predict(test_inputs)

    # Convert one-hot encoded predictions to class labels
    y_pred = np.argmax(y_pred, axis=1)

    # Write the predictions to result.csv
    submission = pd.DataFrame({'PassengerId': ids, 'Transported': y_pred})
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
"""

if __name__ == "__main__":
    #main()