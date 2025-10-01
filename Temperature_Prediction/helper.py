import pickle
import numpy as np
import matplotlib.pyplot as plt


def read_stdata(data_path):
    with open(data_path, 'rb') as file:
        datast = pickle.load(file)
    datast.columns = [ 'day',   'lt',  'obs', 'hres'] + [ str(i) for i in np.arange(51)]
    return datast


def my_train_test_split(data, date):
    train_set = data[data['day'] < date]
    test_set = data[data['day'] >= date]

    return train_set, test_set

def plot_ens_q(data, qs):
    """

    :param data: The data for a SINGLE DAY
    :param qs: The desired range of the quantiles.: E.g.: qs=[0.5] then quantiles will be between 0.25 and 0.75
    :return:
    """
    s = np.arange(6,121, step=6)
    a = np.linspace(0.2,0.6, len(qs))
    k = 0
    for q in qs:
        q1 = data.quantile(0.5 - q/ 2.0, axis=1)
        q2 = data.quantile(0.5 + q / 2.0, axis=1)
        plt.fill_between(s,q1,q2, alpha=a[k])
        k+=1

    plt.xticks(s)
    plt.show()

def plot_verification_rank_histogram(data):
    """

    :param data:
    :return:
    """
    f = [str(i) for i in range(51)]
    dfr = data[['obs'] + f]
    datarank = dfr.rank(axis=1)
    plt.hist(datarank['obs'], density=True)
    plt.plot([0,51], [1.0/51.0, 1.0/51.0], c='r', ls='--')
    plt.ylim((0,0.16))
    plt.show()

def calculate_crps(dataset, save=False):
    f = [str(i) for i in range(51)]

    crps = em.ens_crps(dataset['obs'].to_numpy(),dataset[f].to_numpy())

    if save:
        dfcrps = pd.DataFrame({'day': dataset['day'], 'lt': dataset['lt'], 'crps': crps['crps']})
        dfcrps.to_excel("crps.xlsx")

    return crps['crpsMean']

def crps_lossS(y_true, y_pred):
    """
    Google the exact equation to understand this loss function from hell.

    THIS IS NOT THE FINAL LOSS FUNCTION, WE NEED TO REWRITE IT TO BE TENSORFLOW COMPATIBLE
    :param y_true: The observations we are trying to predict
    :param y_pred: An array with 2 columns: The mean and the standard deviation of the probability distribution
    :return:
    """
    loc = y_pred[:,0]
    sd = y_pred[:,1]
    z = (y_true - loc) / sd
    part1 = z * (2*norm.cdf(z)-1)
    part2 = 2*norm.pdf(z)
    part3 = (1 / np.sqrt(np.pi))

    return sd * (part1 + part2 - part3)

def crps_loss(y_true, y_pred):
    """
    Continuous Ranked Probability Score (CRPS) loss for Gaussian distribution
    y_true: tensor of shape (batch,)
    y_pred: tensor of shape (batch, 2), where
            y_pred[:, 0] = mean (loc)
            y_pred[:, 1] = std (scale)
    """
    # print(type(y_true), type(y_pred[:,0]))
    y_true = torch.squeeze(y_true)
    loc = torch.squeeze(torch.tensor(y_pred[:, 0], dtype=torch.float32))
    sd = torch.squeeze(torch.tensor(y_pred[:, 1], dtype=torch.float32))

    # Standard normal distribution
    dist = torch.distributions.Normal(0.0, 1.0)

    # Standardized residual
    z = (y_true - loc) / sd

    part1 = z * (2 * dist.cdf(z) - 1)
    part2 = 2 * torch.exp(dist.log_prob(z))   # PDF = exp(log_prob)
    part3 = 1.0 / math.sqrt(math.pi)

    return torch.mean(sd * (part1 + part2 - part3))