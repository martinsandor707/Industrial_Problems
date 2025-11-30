import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from models import BinaryClassifier
import helper

df = pd.read_excel("c2k_dataM.xlsx")
# x = helper.create_data(df, ['e']+['p']*15 )
x = helper.create_data(df, ['p']*16 )

print(x.columns)

y = pd.read_csv('isDelayed.csv')
y.rename(columns={y.columns[0]:"nr"}, inplace=True)
y.set_index(y.columns[0], inplace=True)
y.rename(columns={y.columns[0]:"isDelayed"}, inplace=True)
y = y.replace("True", 1)
y = y.replace("False", 0)
y = y.astype(int)
# print(y.iloc[0][0], type(y.iloc[0][0]))
print(y.columns)



# Example setup
method_to_use = "RandomForestClassifier"



if method_to_use == "PytorchBinaryClassifier":

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    input_dim = len(x.columns)  # number of features
    model = BinaryClassifier(input_dim)
    criterion = nn.BCELoss()  # Binary Cross Entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Example training data
    x_train = torch.tensor(x.values, dtype=torch.float32)
    y_train = torch.tensor(y.values, dtype=torch.float32)

    # Training loop
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()

        outputs = model(x_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            preds = (outputs > 0.5).float()
            acc = (preds == y_train).float().mean()
            print(f"Epoch [{epoch + 1}/50], Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")

elif method_to_use == "RandomForestClassifier":
    # Define parameter grid
    param_grid = {
        'n_estimators': [10, 20, 50, 300],
        'max_depth': [None, 5, 10],
        'max_features': [None, 0.1],
        'max_samples': [0.8, 1.0, None]
    }

    # Prepare combinations
    param_combinations = list(itertools.product(
        param_grid['n_estimators'],
        param_grid['max_depth'],
        param_grid['max_features'],
        param_grid['max_samples']
    ))

    results = []

    # Iterate over parameter combinations
    for n_estimators, max_depth, max_features, max_samples in param_combinations:
        acc_scores = []

        # Perform 10 independent runs for each combination
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=i
            )

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                max_samples=max_samples,
                random_state=i,
                n_jobs=-1
            )

            model.fit(X_train, y_train.to_numpy().ravel())
            preds = model.predict(X_test)
            acc = accuracy_score(y_test.to_numpy().ravel(), preds)
            acc_scores.append(acc)

        avg_acc = np.mean(acc_scores)
        results.append({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'max_features': max_features,
            'max_samples': max_samples,
            'avg_accuracy': avg_acc
        })
        print(f"Done: {n_estimators}, {max_depth}, {max_features}, {max_samples} -> {avg_acc:.4f}")

    # Sort by best average accuracy
    results_sorted = sorted(results, key=lambda x: x['avg_accuracy'], reverse=True)

    print("\nTop 5 parameter combinations:")
    for res in results_sorted[:5]:
        print(res)

    top_result = results_sorted[0]

    final_model =RandomForestClassifier(
                n_estimators=top_result['n_estimators'],
                max_depth=top_result['max_depth'],
                max_features=top_result['max_features'],
                max_samples=top_result['max_samples'],
                random_state=42,
                n_jobs=-1
            )

    # Split data once more for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()

    # Train and predict probabilities
    final_model.fit(X_train, y_train)
    y_prob = final_model.predict_proba(X_test)[:, 1]  # probability of positive class

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    print(f"AUC: {auc_score:.4f}")

    # Plot ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for Best Random Forest Model")
    plt.legend()
    plt.grid(True)
    plt.savefig("roc_curve.png")
    plt.show()