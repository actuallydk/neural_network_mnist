import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("Loading MNIST dataset...")
# Download MNIST
data = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = data['data'], data['target'].astype(np.uint8)

# Normalize pixel values
X = X / 255.0

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Remove data augmentation logic and usage
# def augment_data(X, y, n_augment=1000):
#     augmented_X = []
#     augmented_y = []
    
#     for _ in range(n_augment):
#         idx = np.random.randint(0, len(X))
#         img = X[idx].reshape(28, 28)
        
#         # Random rotation (-15 to 15 degrees)
#         angle = np.random.uniform(-15, 15)
#         from scipy.ndimage import rotate
#         img_rotated = rotate(img, angle, reshape=False, mode='nearest')
        
#         # Random translation
#         dx, dy = np.random.randint(-2, 3, 2)
#         from scipy.ndimage import shift
#         img_translated = shift(img_rotated, (dy, dx), mode='nearest')
        
#         augmented_X.append(img_translated.flatten())
#         augmented_y.append(y[idx])
    
#     return np.array(augmented_X), np.array(augmented_y)

# print("Augmenting training data...")
# X_aug, y_aug = augment_data(X_train, y_train, n_augment=2000)
# X_train_aug = np.vstack([X_train, X_aug])
# y_train_aug = np.hstack([y_train, y_aug])

# print(f"Training set size: {len(X_train_aug)}")
# print(f"Test set size: {len(X_test)}")

# Use X_train and y_train directly for training
X_train_aug = X_train
y_train_aug = y_train

print(f"Training set size: {len(X_train_aug)}")
print(f"Test set size: {len(X_test)}")

# Train improved model with better hyperparameters
print("Training neural network...")
clf = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # Deeper network
    activation='relu',
    solver='adam',
    alpha=0.001,  # L2 regularization
    batch_size=128,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=50,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    verbose=True,
    random_state=42
)

clf.fit(X_train_aug, y_train_aug)

# Evaluate
print("\nEvaluating model...")
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")

# Cross-validation score
cv_scores = cross_val_score(clf, X_train_aug[:5000], y_train_aug[:5000], cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, 'mnist_model.pkl')
print("\nModel saved as mnist_model.pkl")