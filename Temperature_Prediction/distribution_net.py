import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from helper import read_stdata, read_stdata2, my_train_test_split, crps_loss, calculate_crps
from MyNN import ProbabilisticRegressor, EarlyStopping, EnsembleTemperatureForecastNN
import torch.nn as nn
import torch.optim

datast1 = read_stdata("t2_station1.pickle")
datast1.iloc[:,2:] = datast1.iloc[:,2:].apply(lambda x: x-273.15)

wind1 = read_stdata2("wind10_hres_station1.pickle")

wind1['wspeed'] = np.sqrt(np.square(wind1['u10'])+np.square(wind1['v10']))
# print(wind1)

tcc1 = read_stdata2("tcc_hres_station1.pickle")
tcc1.columns = ['day', 'lt', 'tcc']

train_1, test_1 = my_train_test_split(datast1, "2018-01-01")
windtrain_1, windtest_1 = my_train_test_split(wind1, "2018-01-01")
tcctrain_1, tcctest_1 = my_train_test_split(tcc1, "2018-01-01")

# print(tcc1)
f = [str(i) for i in range(51)]
train_m1 = pd.DataFrame({'day': train_1['day'], 'lt': train_1['lt'], 'hres': train_1['hres'],
                         'mean': np.mean(train_1[f], axis=1), 'sd': np.std(train_1[f], axis=1),
                         'tcc': tcctrain_1['tcc'], 'wind': windtrain_1['wspeed'],
                         'obs': train_1['obs']
                         })

test_m1 = pd.DataFrame({'day': test_1['day'], 'lt': test_1['lt'], 'hres': test_1['hres'],
                         'mean': np.mean(test_1[f], axis=1), 'sd': np.std(test_1[f], axis=1),
                         'tcc': tcctest_1['tcc'], 'wind': windtest_1['wspeed'],
                         'obs': test_1['obs']
                         })

# print(train_m1)
# print(train_m1.columns)
model = EnsembleTemperatureForecastNN(5, [16])

inputs = torch.tensor(train_m1.iloc[:,2:-1].values, dtype=torch.float32)
targets =  torch.tensor(train_m1['obs'], dtype=torch.float32)

validation_inputs = torch.tensor(train_m1.iloc[:,2:-1].values, dtype=torch.float32)
validation_targets =  torch.tensor(train_m1['obs'], dtype=torch.float32)

output = model(inputs)
print(f"the output shape is {output.size()} \n{output}")
# print(mean_pred.shape, std_pred.shape)


optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

early_stopping = EarlyStopping(patience=5, path="best_model.pt")

epoch_number = 1500
for epoch in range(epoch_number):
    ### TRAINING ###
    model.train()

    # Forward pass
    outputs = model(inputs)
    # print(np.array(outputs).shape)
    # print(type(targets), type(outputs))
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
        print(f"Early stopping triggered")
        print(f'Epoch [{epoch}/{epoch_number}], train_loss: {loss.item():.4f}\tval_loss={val_loss:.4f}')
        break

# restore best weights
model.load_state_dict(torch.load("best_model.pt"))

import matplotlib.pyplot as plt

# preds is assumed to be of shape (N, 2): [mean, std]
preds_np = preds.detach().cpu().numpy()
mean_pred = preds_np[:30, 0]
std_pred = preds_np[:30, 1]
validation_targets_np = validation_targets.detach().cpu().numpy()[:30]
# Calculate quartiles assuming normal distribution
q1 = mean_pred + std_pred * torch.distributions.Normal(0, 1).icdf(torch.tensor(0.25)).item()
q3 = mean_pred + std_pred * torch.distributions.Normal(0, 1).icdf(torch.tensor(0.75)).item()

x = range(len(mean_pred))

plt.figure(figsize=(12, 6))
plt.plot(x, mean_pred, label='Predicted Mean')
plt.plot(x, validation_targets_np, '--', color="red", label='True Values')
plt.fill_between(x, q1, q3, color='skyblue', alpha=0.4, label='Interquartile Range (Q1-Q3)')
plt.xlabel('Sample')
plt.ylabel('Temperature')
plt.title('Predicted Mean and Interquartile Range')
plt.legend()
plt.savefig("predicted_mean_iqr.png")
plt.show()