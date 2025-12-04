import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch.optim.lr_scheduler as lr_scheduler
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


import xgboost as xgb
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
method_to_use = "XGBoostClassifier"


# Best accuracy so far: 0.772
#                  AUC: 0.713
if method_to_use == "PytorchBinaryClassifier":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    batch_size = 64

    # 50-25-25 train-validation-test split
    x_train, x_test, y_train, y_test             = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

    x_train         = scaler.fit_transform(x_train)
    x_validation    = scaler.transform(x_validation)
    x_test          = scaler.transform(x_test)

    ########## DATASET AND DATALOADER ##########
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32)
        # torch.tensor(np.asarray(y_train), dtype=torch.float32).to(device)
    )

    validation_dataset = TensorDataset(
        torch.tensor(x_validation, dtype=torch.float32),
        torch.tensor(y_validation.values, dtype=torch.float32)
    )

    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.float32)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ########## MODEL, LOSS FUNCTION, OPTIMIZER, SCHEDULER, EARLY STOPPING ##########

    input_dim       = len(x.columns)  # number of features
    model           = BinaryClassifier(input_dim).to(device)
    criterion       = nn.BCEWithLogitsLoss()  # Binary Cross Entropy
    optimizer       = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler       = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    early_stopping  = EarlyStopping(patience=5, path="best_model.pt")

    # Storage for plotting
    train_losses = []
    val_losses = []
    test_losses = []

    n_epochs = 250

    # Training loop
    for epoch in range(n_epochs):

        ########## TRAINING ##########
        model.train()
        running_train_loss = 0.0

        for xb, yb in train_dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(xb)
            loss    = criterion(outputs, yb)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() 

        epoch_train_loss = running_train_loss / len(train_dataloader.dataset)
        train_losses.append(epoch_train_loss)
        ########## VALIDATION ##########
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in validation_dataloader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_outputs = model(xb)
                val_loss    = criterion(val_outputs, yb)

                running_val_loss += val_loss.item() 
                preds = (torch.sigmoid(val_outputs) > 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.size(0)

            epoch_val_loss = running_val_loss / len(validation_dataloader.dataset)
            val_losses.append(epoch_val_loss)

            val_acc = correct / total

            if (epoch + 1) % 10 == 0:
                print(  f"[{epoch+1}/{n_epochs}]  "
                        f"Train loss={epoch_train_loss:.4f}   "
                        f"Val loss={epoch_val_loss:.4f}   "
                        f"Val acc={val_acc:.4f}")
        
            running_test_loss = 0.0
            for xb, yb in test_dataloader:
                xb = xb.to(device)
                yb = yb.to(device)
                test_outputs = model(xb)
                test_loss    = criterion(test_outputs, yb)

                running_test_loss += test_loss.item()
            epoch_test_loss = running_test_loss / len(test_dataloader.dataset)
            test_losses.append(epoch_test_loss)

        ########## EARLY STOPPING CHECK ##########
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered")
            print(f'Epoch [{epoch}/{n_epochs}], train_loss: {epoch_train_loss:.4f}\tval_loss={epoch_val_loss:.4f}')
            break

        scheduler.step()


    # Evaluate on test set
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()

    running_test_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():

        for xb, yb in test_dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(xb)
            test_loss    = criterion(outputs, yb)

            running_test_loss += test_loss.item() 
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)

            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    test_loss = running_test_loss / len(test_dataloader.dataset)
    test_acc = correct / total

    print(f"Test Accuracy: {test_acc*100:.2f}%\tTest Loss: {test_loss:.4f}")

    # Compute ROC curve and AUC
    all_logits = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)


    # AUC
    fpr, tpr, _ = roc_curve(all_targets, all_logits)
    auc_score = roc_auc_score(all_targets, all_logits)

    print(f"AUC: {auc_score:.4f}")

    # -----------------------------------------------------------
    # PLOT ROC CURVE
    # -----------------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("charts/roc_curve_nn.png")

    # -----------------------------------------------------------
    # PLOT TRAIN / VAL / TEST LOSSES
    # -----------------------------------------------------------
    plt.figure(figsize=(7, 5))
    epochs_range = range(1, len(train_losses) + 1)

    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.plot(epochs_range, test_losses, label="Test Loss")
    # The test loss is only one number — plot as horizontal line
    # plt.axhline(test_loss, color='red', linestyle='--', label="Test Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training, Validation, and Test Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("charts/loss_curves.png")

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
    plt.savefig("charts/roc_curve_random_forest.png")
    plt.show()


# Best accuracy so far: 0.789 
#                  AUC: 0.803
# Params:
# {'colsample_bytree': 0.8, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 200, 'subsample': 1.0}
elif method_to_use == "XGBoostClassifier":

    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 3, 5]
    }

    xgb_clf = xgb.XGBClassifier(random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',    # or another metric
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("Best params:", grid_search.best_params_)
    print("Best average accuracy:", grid_search.best_score_)

    final_model = grid_search.best_estimator_

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
    plt.title(f"ROC Curve for Best XGBoost Model")
    plt.legend()
    plt.grid(True)
    plt.savefig("charts/roc_curve_xgb.png")