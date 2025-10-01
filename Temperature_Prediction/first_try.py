import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from helper import read_stdata, my_train_test_split, plot_ens_q
from MyNN import TemperatureRegressor
import torch.nn as nn
import torch.optim

datast1 = read_stdata("t2_station1.pickle")
datast1.iloc[:,2:] = datast1.iloc[:,2:].apply(lambda x: x-273.15)  # Convert temperature from Kelvin to Celsius
# print(datast1.head())

# 2017-12-31 is the halfway point

train, test = my_train_test_split(datast1, "2018-01-01")
ds0 = train[train['day'] =="2017-12-31"] # Visualizing the spread of perturbed predictions
plot_ens_q(ds0.iloc[:,4:], [0.8, 0.5])

print(train)

mse_hres = mean_squared_error(train['obs'], train['hres'])

mse_0 = mean_squared_error(train['obs'], train['0'])

perturbation_means = np.mean(train.iloc[:,4:], axis=1)

mse_pert_means = mean_squared_error(train['obs'], perturbation_means)

print(f"Hres mean:\t{mse_hres}\nMse_0:\t{mse_0}\nMse_pert:\t{mse_pert_means}")


model = TemperatureRegressor(51, 126)



inputs = torch.tensor(train.iloc[:,4:].values, dtype=torch.float32)
targets =  torch.tensor(train['obs'], dtype=torch.float32)

temperature_preds = model(inputs)
print(temperature_preds.shape)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

epoch_number = 1500
for epoch in range(epoch_number):
    model.train()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{epoch_number}], Loss: {loss.item():.4f}')