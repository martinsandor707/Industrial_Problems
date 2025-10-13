import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from helper import read_stdata, my_train_test_split, crps_loss, calculate_crps, crps_lossS, coverage_ens_nominal, coverage_ens, coverage_pred
from MyNN import ProbabilisticRegressor, EarlyStopping, EnsembleTemperatureForecastNN
import torch.nn as nn
import torch.optim

datast1 = read_stdata("t2_station1.pickle")
datast1.iloc[:,2:] = datast1.iloc[:,2:].apply(lambda x: x-273.15)

train_1, test_1 = my_train_test_split(datast1, "2018-01-01")
train_1, validation_1 = my_train_test_split(train_1, "2017-07-01")
print(f"{train_1.columns=}")

f = [str(i) for i in range(51)]



model = EnsembleTemperatureForecastNN(51, hidden_sizes=[16])

inputs = torch.tensor(train_1.iloc[:,4:].values, dtype=torch.float32)
targets =  torch.tensor(train_1['obs'].values, dtype=torch.float32)

validation_inputs = torch.tensor(validation_1.iloc[:,4:].values, dtype=torch.float32)
validation_targets =  torch.tensor(validation_1['obs'].values, dtype=torch.float32)

test_inputs = torch.tensor(test_1.iloc[:,4:].values, dtype=torch.float32)
test_targets =  torch.tensor(test_1['obs'].values, dtype=torch.float32)


output = model(inputs)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

early_stopping = EarlyStopping(patience=50, path="best_model.pt")

epoch_number = 10000
for epoch in range(epoch_number):
    ### TRAINING ###
    model.train()

    # Forward pass
    outputs = model(inputs)
    loss = crps_loss(targets, outputs)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



    ### VALIDATION ###
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        preds = model(validation_inputs)
        val_loss += crps_loss(validation_targets, preds ).item()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{epoch_number}], train_loss: {loss.item():.4f}\tval_loss={val_loss:.4f}')

    ### EARLY STOPPING CHECK ###
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        test_preds = model(test_inputs)
        test_loss = crps_loss(test_targets, test_preds)
        print(f"Early stopping triggered")
        print(f'Epoch [{epoch}/{epoch_number}], train_loss: {loss.item():.4f}\tval_loss={val_loss:.4f}')
        break

# Restore best weights
# Best results so far: val_loss = 1.1494     test_loss = 1.3150
# Hidden layer: 16
# Learning rate: 1e-3
# Patience: 50
# Epochs until early stop triggered: 827
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
test_preds = model(test_inputs)
test_loss = crps_loss(test_targets, test_preds)
print(f'test_loss={test_loss:.4f}')

preds = model(test_inputs).detach().cpu().numpy()

import matplotlib.pyplot as plt

# preds is assumed to be of shape (N, 2): [mean, std]
preds_np = preds
print(f"{preds=}")
# mean_pred = preds_np[:30, 0]
# std_pred = preds_np[:30, 1]
# test_targets_np = test_targets.detach().cpu().numpy()[:30]
mean_pred = preds_np[-30:, 0]
std_pred = preds_np[-30:, 1]
test_targets_np = test_targets.detach().cpu().numpy()[-30:]
# Calculate quartiles assuming normal distribution
q1 = mean_pred + std_pred * torch.distributions.Normal(0, 1).icdf(torch.tensor(0.25)).item()
q3 = mean_pred + std_pred * torch.distributions.Normal(0, 1).icdf(torch.tensor(0.75)).item()

x = range(len(mean_pred))

plt.figure(figsize=(12, 6))
plt.plot(x, mean_pred, label='Predicted Mean')
plt.plot(x, test_targets_np, '--', color="red", label='True Values')
plt.fill_between(x, q1, q3, color='skyblue', alpha=0.4, label='Interquartile Range (Q1-Q3)')
plt.xlabel('Sample')
plt.ylabel('Temperature')
plt.title('Predicted Mean and Interquartile Range')
plt.legend()
plt.savefig("predicted_mean_iqr.png")
# plt.show()

################# CRPS MEANS


cval = crps_lossS(test_targets.detach().cpu().numpy(), preds)
distcrps = pd.DataFrame({'day': test_1['day'], 'lt': test_1['lt'], 'crps': cval})
distcrps.to_excel("dist_crps.xlsx")

print(f"Mean of cval: {np.mean(cval)}")

dfres1 = pd.DataFrame({'day': test_1['day'], 'lt': test_1['lt'], 'loc': preds[:,0], 'scale': preds[:,1], 'crps': cval})
distcrps.to_excel("output_prob1.xlsx")

errnew2 = mean_squared_error(test_1['obs'], preds[:,0])

a = []
b = []

for lt in range(1, 21):
    ds = distcrps[distcrps['lt'] == lt]
    # << Calculate ensemble crps here >>
    ds2 = test_1[test_1['lt'] == lt]
    a.append(np.mean(ds['crps']))
    b.append(np.mean(calculate_crps(ds2)['crps']))


plt.figure(figsize=(12, 6))
plt.plot(range(1,21), a, color="blue", label='Predicted cprs')
plt.plot(range(1,21), b, color="red",  label='Raw Ensemble cprs mean')
plt.xlabel('Lead time')
plt.ylabel('CRPS')
plt.title('Predicted CRPS and raw ensemble CRPS')
plt.legend()
plt.savefig("mean_crps.png")
# plt.show()

###### COVERAGE
alpha = 0.2
cov_ens_nom = coverage_ens_nominal(test_1)[0]
cov_ens = coverage_ens(test_1, alpha)[0]
cov_pred = coverage_pred(preds_np, test_1, alpha)[0]
# print(f"{cov_ens=}\n{cov_pred=}")
plt.figure(figsize=(12, 6))
plt.plot(range(1,21), cov_pred, color="blue", label='Predicted Coverage')
plt.plot(range(1,21), cov_ens, color="red", label='Raw Ensemble Coverage')
plt.plot(range(1,21), cov_ens_nom, "--", color="red",  label='Nominal Coverage')
plt.xlabel('Lead time')
plt.ylabel('Coverage')
plt.title('Predicted Coverage and raw ensemble Coverage')
plt.legend()
plt.savefig("coverage.png")
# plt.show()


####### WIDTH
width_ens = coverage_ens(test_1, alpha)[1]
width_pred = coverage_pred(preds_np, test_1, alpha)[1]
# print(f"{cov_ens=}\n{cov_pred=}")
plt.figure(figsize=(12, 6))
plt.plot(range(1,21), width_pred, color="blue", label='Predicted Width')
plt.plot(range(1,21), width_ens, color="red", label='Raw Ensemble Width')
plt.xlabel('Lead time')
plt.ylabel('Width')
plt.title('Predicted Width and raw ensemble Width')
plt.legend()
plt.savefig("width.png")