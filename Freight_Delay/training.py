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

from models import BinaryClassifier, EarlyStopping
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
method_to_use = "PytorchBinaryClassifier"



if method_to_use == "PytorchBinaryClassifier":

    # 50-25-25 train-validation-test split
    x_train, x_test, y_train, y_test             = train_test_split(x, y, test_size=0.25, random_state=42)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

    input_dim       = len(x.columns)  # number of features
    model           = BinaryClassifier(input_dim)
    criterion       = nn.BCELoss()  # Binary Cross Entropy
    optimizer       = torch.optim.Adam(model.parameters(), lr=1e-3)
    early_stopping  = EarlyStopping(patience=10, path="best_model.pt")


    # Example training data
    x_train      = torch.tensor(x_train.values, dtype=torch.float32)
    x_validation = torch.tensor(x_validation.values, dtype=torch.float32)
    y_train      = torch.tensor(y_train.values, dtype=torch.float32)
    y_validation = torch.tensor(y_validation.values, dtype=torch.float32)

    n_epochs = 250

    # Training loop
    for epoch in range(n_epochs):

        ########## TRAINING ##########
        model.train()

        outputs = model(x_train)
        loss    = criterion(outputs, y_train)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        ########## VALIDATION ##########
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_outputs = model(x_validation)
            val_loss    = criterion(val_outputs, y_validation)

        if (epoch + 1) % 10 == 0:
            preds = (outputs > 0.5).float()
            val_preds = (val_outputs > 0.5).float()
            acc = (preds == y_train).float().mean()
            val_acc = (val_preds == y_validation).float().mean()
            print(f"Epoch [{epoch + 1}/{n_epochs}]\n\tTrain_loss: {loss.item():.4f},\tAccuracy: {acc.item():.4f}\n\tVal_loss: {val_loss.item():.4f}\tVal_Accuracy: {val_acc.item():.4f}\n##########")

        ########## EARLY STOPPING CHECK ##########
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered")
            print(f'Epoch [{epoch}/{n_epochs}], train_loss: {loss.item():.4f}\tval_loss={val_loss:.4f}')
            break

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(x_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32)
        outputs = model(x_test)
        preds = (outputs > 0.5).float()
        acc = (preds == y_test).float().mean()
        print(f"Test Accuracy: {acc.item()*100:.2f}%")

    # Compute ROC curve and AUC
    y_prob = outputs.numpy()
    fpr, tpr, _ = roc_curve(y_test.numpy(), y_prob)
    auc_score = roc_auc_score(y_test.numpy(), y_prob)

    print(f"AUC: {auc_score:.4f}")

    # Plot ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for Neural Network (AUC = {auc_score:.3f})")
    plt.legend()
    plt.grid(True)
    plt.savefig("roc_curve_nn.png")


# Best accuracy so far: 0.786 with n_estimators=300, max_depth=None, max_features=0.1, max_samples=0.8
#                  AUC: 0.783
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
    plt.savefig("roc_curve_random_forest.png")
    plt.show()