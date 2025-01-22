import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import random
from numpy import linalg as la
from sklearn import linear_model as lm
import math
import time
from math import sqrt
from scipy import linalg
from bl_bandit import ista
import torch


def one_hot_encode(features_to_encode, dataset):
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(dataset[features_to_encode])

    encoded_cols = pd.DataFrame(encoder.transform(dataset[features_to_encode]), columns=encoder.get_feature_names())
    dataset = dataset.drop(columns=features_to_encode)
    dataset = np.array(dataset, dtype=float)
    encoded_cols = np.array(encoded_cols, dtype=float)
    dataset = np.concatenate((dataset, encoded_cols), axis=1)
    return dataset


def data_loader(center_id=None):
    train_df = pd.read_csv('./data/price/train.csv')
    meal_df = pd.read_csv('./data/price/meal_info.csv')
    # center_df = pd.read_csv('./data/price/fulfilment_center_info.csv')
    # train_df = train_df.merge(center_df, left_on='center_id', right_on='center_id', how="left")
    train_df = train_df.merge(meal_df, left_on='meal_id', right_on='meal_id', how="left")
    if center_id is not None:
        train_df_c = pd.DataFrame(columns=train_df.columns)
        for center in center_id:
            train_df_c = pd.concat([train_df_c, train_df[train_df['center_id'] == center]], axis=0)

    train_df = train_df_c
    train_df['num_orders'] = train_df['num_orders'].astype('int')
    train_df['checkout_price'] = train_df['checkout_price'].astype('int')

    train_df.describe()

    price_true = train_df['checkout_price']
    demand_true = np.array(train_df['num_orders'], dtype=float)
    demand_true = demand_true + np.random.randn(demand_true.shape[0])
    train_df = train_df.drop(['id', 'center_id', 'meal_id', 'week', 'checkout_price', 'num_orders'],
                             axis=1)
    price_min = min(train_df['base_price'])
    price_max = max(train_df['base_price'])
    train_df['base_price'] = (train_df['base_price'] - price_min) / (price_max - price_min)
    # one hot encode
    features_to_encode = ['category', 'cuisine']
    train_df = one_hot_encode(features_to_encode, train_df)
    per = np.random.permutation(train_df.shape[0])
    train_df = train_df[per]
    demand_true = demand_true[per]
    train_df = np.concatenate((train_df, np.ones((train_df.shape[0],1))), axis=1)

    return train_df, np.array(price_true, dtype=float), demand_true


def oracle_esti(centers, lam=1, device='cpu'):
    train_df, price_true, demand_true = data_loader(centers)
    # oracle beta estimaiton
    X = np.array(train_df, dtype=float)
    XX = np.concatenate((X, np.multiply(X.T, price_true).T), axis=1)

    """XX = torch.tensor(XX, device=device)
    demand_true = torch.tensor(demand_true, device=device)
    lam = torch.tensor(lam, dtype=float, device=device)"""

    reg_lasso_2 = lm.Lasso(alpha=lam, fit_intercept=False, max_iter=1e6)
    reg_lasso_2.fit(XX, demand_true)
    beta = reg_lasso_2.coef_

    """beta, _, _ = ista(XX, demand_true, lam, 100000, device)
    XX = XX.cpu().numpy()
    demand_true = demand_true.cpu().numpy()"""

    # OLS
    # beta = np.linalg.inv((XX.T).dot(XX)).dot((XX.T).dot(demand_true.reshape(-1,1)))

    pre_error = np.linalg.norm((XX.dot(beta)).reshape(-1,1) - demand_true.reshape(-1,1)) / XX.shape[0]
    print("prediction error: %10.5f" % pre_error)
    beta.tofile("beta_oracle_cpu.bin")
    return beta


if __name__ == '__main__':
    print(torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    centers = [10, 13, 41, 43, 53, 91]
    #train_data, price_true, demand_true = data_loader([13])
    beta_true = oracle_esti(centers, 0.0001)
    print(beta_true)
