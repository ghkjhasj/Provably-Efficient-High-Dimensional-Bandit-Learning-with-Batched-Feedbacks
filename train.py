import numpy as np
import matplotlib.pyplot as plt
import argparse
from bl_bandit import *
from blr_bandit import *
import random
import torch
import os
import csv
# import warfarin_data_loader as ldl
from price_data_loader import data_loader, oracle_esti


def train_toy_blb(args):
    """ Batched LASSO Bandit """
    print("Batched LASSO Bandit toy:")
    random.seed(args.seed)
    # initialize bandit
    blBandit = BatchedLassoBandit(args.data_dim, args.arm_num, args.h, args.t_0, args.lam_1, args.lam_2_0, args.c)

    # true regression coefficient for toy data
    beta_true = [np.random.randn(args.data_dim, 1) for _ in range(args.arm_num)]
    # make them sparse
    for a in range(args.arm_num):
        idx = np.random.choice(range(args.data_dim), args.data_dim - args.sparse, replace=False)
        beta_true[a][idx] = 0

    # save rewards and actions
    rew_opt = np.zeros((args.time_steps, 1))
    rew_blb = np.zeros((args.time_steps, 1))
    act_blb = np.zeros(args.time_steps, dtype=int)
    act_opt = np.zeros(args.time_steps, dtype=int)

    # get grid
    grid = blBandit.choose_grid(args.time_steps, args.batches)  # from 1 to T
    print(grid)
    L = len(grid)
    print(L)
    X = np.zeros((args.time_steps, args.data_dim), dtype=float)

    for l in range(L):
        # print('Batch %d' % (l+1))
        if l == 0:
            t_0 = 0
        else:
            t_0 = grid[l - 1]
        t_1 = grid[l]
        for t in range(t_0, t_1):
            # observe context truncate at 1/-1
            x = np.random.randn(args.data_dim)
            x[x > 1] = 1
            x[x < -1] = -1
            X[t] = x

            # find bandit action
            act_blb[t] = blBandit.get_opt_action(x, t)
            act_opt[t] = np.argmax([beta_true[a].T.dot(x) for a in range(args.arm_num)])

            # compute reward but reveal at the end of this batch
            noise = np.random.randn(1) * 0.01
            rew_blb[t] = beta_true[act_blb[t]].T.dot(x) + noise
            rew_opt[t] = beta_true[act_opt[t]].T.dot(x) + noise

        # update bandit parameters
        blBandit.update_est(X[:t_1], rew_blb[:t_1], t_1)

    return np.cumsum(rew_opt - rew_blb)


def train_wfr_blb(args):
    """ Batched LASSO Bandit """
    print("Batched LASSO Bandit warfarin:")
    random.seed(args.seed)
    # get data
    X, dose_true = get_warfarin()
    # initialize bandit
    blBandit = BatchedLassoBandit(args.data_dim, args.arm_num, args.h, args.t_0, args.lam_1, args.lam_2_0, args.c)

    # save rewards and actions
    rew_blb = np.zeros((args.time_steps, 1))
    act_blb = np.zeros(args.time_steps, dtype=int)

    # get grid
    grid = blBandit.choose_grid(args.time_steps, args.batches)  # from 1 to T
    print(grid)
    L = len(grid)
    print(L)

    for l in range(L):
        print(l + 1)
        if l == 0:
            t_00 = 0
        else:
            t_00 = grid[l - 1]
        t_1 = grid[l]
        for t in range(t_00, t_1):
            x = X[t]
            # find bandit action
            act_blb[t] = blBandit.get_opt_action(x, t)
            # compute reward but reveal at the end of this batch
            rew_blb[t] = 0 if act_blb[t] == dose_true[t] else -1

        # update bandit parameters
        blBandit.update_est(X[:t_1], rew_blb[:t_1], t_1)

    return -np.cumsum(rew_blb) / np.arange(1, args.time_steps + 1)


def train_toy_lb(args):
    """ LASSO Bandit """
    print("LASSO Bandit toy:")
    random.seed(args.seed)
    # initialize bandit
    lBandit = LassoBandit(args.data_dim, args.arm_num, args.q, args.h, args.lam_1, args.lam_1)

    # true regression coefficient for toy data
    beta_true = [np.random.randn(args.data_dim, 1) for _ in range(args.arm_num)]
    # make them sparse
    for a in range(args.arm_num):
        idx = np.random.choice(range(args.data_dim), args.data_dim - args.sparse, replace=False)
        beta_true[a][idx] = 0

    # save samples, rewards and actions
    rew_opt = np.zeros((args.time_steps, 1))
    rew_lb = np.zeros((args.time_steps, 1))
    act_lb = np.zeros(args.time_steps, dtype=int)
    act_opt = np.zeros(args.time_steps, dtype=int)
    X = np.zeros((args.time_steps, args.data_dim), dtype=float)
    lBandit.forced_sample_set(args.time_steps)

    for t in range(args.time_steps):
        # observe context truncate at 1/-1
        x = np.random.randn(args.data_dim)
        x[x > 1] = 1
        x[x < -1] = -1
        X[t] = x

        # find bandit action
        act_lb[t] = lBandit.get_opt_action(x, t)
        act_opt[t] = np.argmax([beta_true[a].T.dot(x) for a in range(args.arm_num)])

        # compute reward
        noise = np.random.randn(1) * 0.01
        rew_lb[t] = beta_true[act_lb[t]].T.dot(x) + noise
        rew_opt[t] = beta_true[act_opt[t]].T.dot(x) + noise

        # update bandit parameters
        lBandit.update_est(X[:(t + 1)], rew_lb[:(t + 1)], act_lb[t], t)

    return np.cumsum(rew_opt - rew_lb)


def train_wfr_lb(args):
    """ Warfarin LASSO Bandit """
    print("LASSO Bandit warfarin:")
    random.seed(args.seed)
    # get data
    X, dose_true = get_warfarin()
    # initialize bandit
    lBandit = LassoBandit(args.data_dim, args.arm_num, args.q, args.h, args.lam_1, args.lam_1)

    # save rewards and actions
    rew_lb = np.zeros((args.time_steps, 1))
    act_lb = np.zeros(args.time_steps, dtype=int)
    lBandit.forced_sample_set(args.time_steps)

    for t in range(args.time_steps):
        print(t)
        # observe context truncate at 1/-1
        x = X[t]
        # find bandit action
        act_lb[t] = lBandit.get_opt_action(x, t)
        # compute reward but reveal at the end of this batch
        rew_lb[t] = 0 if act_lb[t] == dose_true[t] else -1

        # update bandit parameters
        lBandit.update_est(X[:(t + 1)], rew_lb[:(t + 1)], act_lb[t], t)

    return -np.cumsum(rew_lb) / np.arange(1, args.time_steps + 1)


# Pricing LASSO Bandit
def train_price_lb(args, train_data, price_true, demand_true, beta_true):
    """ Pricing Sequential LASSO Bandit """
    print("LASSO Bandit Pricing:")
    random.seed(args.seed)
    # initialize bandit
    lBandit = Price_LassoBandit(args.data_dim, args.q, args.h, args.lam_2_0, device=device)

    # save samples, rewards and actions
    rew_opt = (price_true * demand_true).reshape(-1, 1)  # demand * price
    # rew_opt = np.zeros((args.time_steps, 1))
    rew_lb = np.zeros((args.time_steps, 1))
    y_lb = np.zeros((args.time_steps, 1))  # demand in lb
    act_lb = np.zeros(args.time_steps, dtype=float)  # action is price

    lBandit.forced_sample_set(args.time_steps)

    for t in range(args.time_steps):
        if (t + 1) % 1000 == 0:
            print('Time %d' % (t + 1))
        # find bandit action
        x_t = train_data[t]
        d = x_t.shape[0]
        act_lb[t] = lBandit.get_opt_action(x_t.reshape(-1, 1), t)

        # compute reward
        noise = np.random.randn(1)
        # xx_t = np.concatenate((x_t, np.multiply(x_t.T, act_lb[t]).T))
        # y_lb[t] = beta_true.T.dot(xx_t.reshape(-1, 1)) + noise
        y_lb[t] = beta_true[:d].T.dot(x_t.reshape(-1, 1)) + act_lb[t] * beta_true[d:].T.dot(x_t.reshape(-1, 1)) + noise
        rew_lb[t] = max(y_lb[t], 0) * act_lb[t]

        """p_opt = -0.5 * (x_t.T.dot(beta_true[:d].reshape(-1, 1))) / (x_t.T.dot(beta_true[d:].reshape(-1, 1)))
        p_opt = min(max(p_opt, 0), 1000)
        xx_t = np.concatenate((x_t, np.multiply(x_t.T, p_opt).T))
        rew_opt[t] = (max(beta_true.T.dot(xx_t.reshape(-1, 1)) + noise, 0))"""

        # update bandit parameters
        if np.sqrt(t + 1) % 1 == 0:
            lBandit.update_est(train_data[:(t + 1)], y_lb[:(t + 1)], act_lb[:(t + 1)], t + 1)

    return np.cumsum(rew_opt - rew_lb)


def train_price_blb(args, train_data, price_true, demand_true, beta_true):
    """ Pricing Batched LASSO Bandit """
    print("Batched LASSO Bandit Pricing:")
    random.seed(args.seed)
    # initialize bandit
    blBandit = Price_BatchedLassoBandit(args.data_dim, args.t_0, args.lam_2_0, args.c, device)

    # save samples, rewards and actions
    rew_opt = (price_true * demand_true).reshape(-1, 1)  # demand * price
    # rew_opt = np.zeros((args.time_steps, 1))
    rew_blb = np.zeros((args.time_steps, 1))
    y_blb = np.zeros((args.time_steps, 1))  # demand
    act_blb = np.zeros(args.time_steps, dtype=float)  # action is price

    # get grid
    grid = blBandit.choose_grid(args.time_steps, args.batches)  # from 1 to T
    grid = grid
    print(grid)
    L = len(grid)
    print(L)

    for l in range(L):
        print('Batch %d' % (l + 1))
        if l == 0:
            t_0 = 0
        else:
            t_0 = grid[l - 1]
        t_1 = grid[l]
        for t in range(t_0, t_1):
            # find bandit action
            x_t = train_data[t]
            d = x_t.shape[0]
            act_blb[t] = blBandit.get_opt_action(x_t.reshape(-1, 1), t)

            # compute reward but reveal at the end of this batch
            noise = np.random.randn(1)
            # xx_t = np.concatenate((x_t, np.multiply(x_t.T, act_blb[t]).T))
            # y_blb[t] = beta_true.T.dot(xx_t.reshape(-1, 1)) + noise
            y_blb[t] = beta_true[:d].T.dot(x_t.reshape(-1, 1)) + act_blb[t] * beta_true[d:].T.dot(
                x_t.reshape(-1, 1)) + noise
            rew_blb[t] = max(y_blb[t], 0) * act_blb[t]

            """p_opt = -0.5 * (x_t.T.dot(beta_true[:d].reshape(-1,1))) / (x_t.T.dot(beta_true[d:].reshape(-1,1)))
            p_opt = min(max(p_opt, 0), 1000)
            xx_t = np.concatenate((x_t, np.multiply(x_t.T, p_opt).T))
            rew_opt[t] = (max(beta_true.T.dot(xx_t.reshape(-1, 1)) + noise, 0)) * p_opt"""

        # update bandit parameters
        blBandit.update_est(train_data[:t_1], y_blb[:t_1], act_blb[:t_1], t_1 + 1)

    return np.cumsum(rew_opt - rew_blb)


# Low-Rank Bandit
def train_lowrank(args):
    """ Batched and Sequential Low-Rank Bandit """
    print("Batched Low-Rank Bandit:")
    # initialize bandit
    blr_bandit = BatchedLowRankBandit(args.d_1, args.d_1, args.arm_num, args.h, args.t_0, args.lam_1, args.lam_2_0,
                                      args.c)

    # true regression coefficient for toy data
    beta_true = [np.diag(np.random.randn(args.d_1)) for _ in range(args.arm_num)]
    # make them sparse
    for a in range(args.arm_num):
        idx = np.random.choice(range(args.d_1), args.d_1 - args.rank, replace=False)
        beta_true[a][idx, idx] = 0

    # save rewards and actions
    rew_opt = np.zeros(args.time_steps)
    rew_blb = np.zeros(args.time_steps)
    act_blb = np.zeros(args.time_steps, dtype=int)
    act_opt = np.zeros(args.time_steps, dtype=int)

    X = np.zeros((args.time_steps, args.d_1, args.d_1), dtype=float)
    print(args.batches)
    # sequential
    if args.batches == args.time_steps:
        print("sequential")
        for t in range(args.time_steps):
            if (t + 1) % 100 == 0:
                print(t + 1)
            # observe context truncate at 1/-1
            x = np.random.randn(args.d_1, args.d_1)
            x[x > 1] = 1
            x[x < -1] = -1
            X[t] = x

            # find bandit action
            act_blb[t] = blr_bandit.get_opt_action(x, t)
            act_opt[t] = np.argmax([np.trace(x.T.dot(beta_true[a])) for a in range(args.arm_num)])

            # compute reward
            noise = np.random.randn(1) * 0.01
            rew_blb[t] = np.trace(x.T.dot(beta_true[act_blb[t]])) + noise
            rew_opt[t] = np.trace(x.T.dot(beta_true[act_opt[t]])) + noise

            # update bandit parameters
            blr_bandit.update_est(X[:(t + 1)], rew_blb[:(t + 1)], t)

        return np.cumsum(rew_opt - rew_blb)
    else:
        # Batched version
        # get grid
        grid = blr_bandit.choose_grid(args.time_steps, args.batches)  # from 1 to T
        print(grid)
        L = len(grid)
        print(L)

        for l in range(L):
            # print('Batch %d' % (l+1))
            if l == 0:
                t_0 = 0
            else:
                t_0 = grid[l - 1]
            t_1 = grid[l]
            for t in range(t_0, t_1):
                # observe context truncate at 1/-1
                x = np.random.randn(args.d_1, args.d_1)
                x[x > 1] = 1
                x[x < -1] = -1
                X[t] = x

                # find bandit action
                act_blb[t] = blr_bandit.get_opt_action(x, t)
                act_opt[t] = np.argmax([np.trace(x.T.dot(beta_true[a])) for a in range(args.arm_num)])

                # compute reward but reveal at the end of this batch
                noise = np.random.randn(1)
                rew_blb[t] = np.trace(x.T.dot(beta_true[act_blb[t]])) + noise
                rew_opt[t] = np.trace(x.T.dot(beta_true[act_opt[t]])) + noise

            # update bandit parameters
            blr_bandit.update_est(X[:t_1], rew_blb[:t_1], t_1)

        return np.cumsum(rew_opt - rew_blb)


def train_LowOFUL(args):
    """ LowOFUL with flattened features for linear bandit """
    print("LowOFUL:")

    # Initialize OFUL Linear Bandit with flattened arms
    # Initialize reward and action storage
    rew_opt = np.zeros(args.time_steps)
    rew_oful = np.zeros(args.time_steps)
    act_oful = np.zeros(args.time_steps, dtype=int)
    act_opt = np.zeros(args.time_steps, dtype=int)
    # True regression coefficient for toy data (flattened and sparse)
    beta_true = [np.diag(np.random.randn(args.d_1)) for _ in range(args.arm_num)]
    # make them sparse
    for a in range(args.arm_num):
        idx = np.random.choice(range(args.d_1), args.d_1 - args.rank, replace=False)
        beta_true[a][idx, idx] = 0
        beta_true[a] = beta_true[a].flatten()

    X = np.zeros((args.time_steps, args.d_1 * args.d_1), dtype=float)
    for t in range(args.time_steps):
        # observe context truncate at 1/-1
        x = np.random.randn(args.d_1, args.d_1)
        x[x > 1] = 1
        x[x < -1] = -1
        x_flat = x.flatten()
        X[t] = x_flat
        act_opt[t] = np.argmax([x_flat @ beta_true[a] for a in range(args.arm_num)])

    beta_true = np.concatenate(beta_true).reshape(args.d_1 *  args.d_1, args.arm_num)

    # arm_set = np.array([np.random.randn(args.d_1 * args.d_1) for _ in range(args.arm_num)])
    oful_bandit = CMMAB(
        T=args.time_steps,
        k=args.arm_num,
        means=beta_true,
        context=X,
        labels=act_opt,
        d=args.d_1 * args.d_1
    )
    L = 10000

    regret, _, _ = oful_bandit.oful(lam=args.lam_1)
    return np.cumsum(regret)

    # if L == args.time_steps:
    #     print("sequential")
    #     for t in range(args.time_steps):
    #         if (t + 1) % 100 == 0:
    #             print(t + 1)

    #         # OFUL action selection and optimal action
    #         act_oful[t] = oful_bandit.next_arm()
    #         print(act_oful[t])

    #         # Reward calculation with added noise
    #         noise = np.random.randn() * 0.01
    #         rew_oful[t] = x_flat @ beta_true[act_oful[t]] + noise
    #         rew_opt[t] = x_flat @ beta_true[act_opt[t]] + noise

    #         # Update OFUL estimates in batch if specified
    #         oful_bandit.update(act_oful[:(t + 1)], rew_oful[:(t + 1)])

    #     # Return the cumulative regret
    #     return np.cumsum(rew_opt - rew_oful)
    # else:
    #     # Batched version
    #     # get grid
    #     grid = np.arange(0, args.time_steps+1, args.time_steps / L).astype(int)
    #     print(grid)

    #     for l in range(L):
    #         # print('Batch %d' % (l+1))
    #         t_0 = grid[l - 1]
    #         t_1 = grid[l]
    #         for t in range(t_0, t_1):
    #             # OFUL action selection and optimal action
    #             act_oful[t] = oful_bandit.get_opt_action(x_flat)
    #             act_opt[t] = np.argmax([x_flat @ beta_true[a] for a in range(args.arm_num)])

    #             # Reward calculation with added noise
    #             noise = np.random.randn() * 0.01
    #             rew_oful[t] = x_flat @ beta_true[act_oful[t]] + noise
    #             rew_opt[t] = x_flat @ beta_true[act_opt[t]] + noise

    #         # Update OFUL estimates in batch if specified
    #         oful_bandit.update_est(X[:(t_1 + 1)], act_oful[:(t_1 + 1)], rew_oful[:(t_1 + 1)])

    #     return np.cumsum(rew_opt - rew_oful)


def get_warfarin():
    file_path = './data/warfarin.csv'
    with open(file_path) as f:
        reader = csv.reader(f, delimiter=',')

        is_header = True
        wfr_data = []

        for row in reader:
            if (is_header):
                # skip the header
                is_header = False
                continue
            wfr_data.append(np.array(row, dtype=float))

    wfr_data = np.array(wfr_data)
    per = np.random.permutation(wfr_data.shape[0])
    X = wfr_data[per, :93]
    # X = X/np.max(np.abs(X), axis=0)
    y = wfr_data[per, 93]
    return X, y


def make_figure(rg_blb, rg_lb=None):
    plt.figure()
    # plt.title("Lasso Bandit")
    plt.plot(rg_blb, label="Batched Lasso Bandit")
    if rg_lb is not None:
        plt.plot(rg_lb, label="Lasso Bandit")
    plt.xlabel("time")
    plt.ylabel("cumulative regret")
    plt.legend()
    plt.savefig('./reg.pdf')
    plt.show()
    plt.close()


def write_config_to_file(config, save_path):
    with open(os.path.join(save_path, 'config.txt'), 'w') as file:
        for arg in vars(config):
            file.write(str(arg) + ': ' + str(getattr(config, arg)) + '\n')


if __name__ == '__main__':

    global deviceD
    print(torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument("--trails", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--dataset", type=str, default='toy_lowrank',
                        choices=['toy_lasso', 'toy_lowrank', 'warfarin', 'price'])
    parser.add_argument("--arm_num", type=int, default=10)
    parser.add_argument("--time_steps", type=int, default=10000)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--t_0", type=int, default=5, help='forced-sampling parameter')
    parser.add_argument("--q", type=int, default=1, help='forced sample parameter for LASSO bandit')
    parser.add_argument("--h", type=float, default=5.0, help='localization parameter')
    parser.add_argument("--lam_1", type=float, default=0.05, help='regularization parameter')
    parser.add_argument("--lam_2_0", type=float, default=0.05, help='initial regularization parameter')
    parser.add_argument("--c", type=float, default=1.01, help='coefficient for a')  # 10: c=3 a=6, 100: c=1.01 a=2.02

    # for high-dim
    parser.add_argument("--data_dim", type=int, default=22)
    parser.add_argument("--sparse", type=int, default=5)

    # for low-rank matrix
    parser.add_argument("--d_1", type=int, default=10, help='d_1=d_2')
    parser.add_argument("--d", type=int, default=10, help='d=d_1')
    parser.add_argument("--rank", type=int, default=2)

    args = parser.parse_args()

    batches = [100]  # , 100, 10000

    if args.dataset == 'toy_lasso':
        save_dir = './result_lasso/toy_d'
        for t in range(args.trails):
            print("trial: %d" % t)
            blb_rg = train_toy_blb(args)
            # lb_rg = train_toy_lb(args)
            torch.save(blb_rg, os.path.join(save_dir, 'blb-rg-' + str(t)))
            # torch.save(lb_rg, os.path.join(save_dir, 'lb-rg-' + str(t)))
            make_figure(blb_rg)
    elif args.dataset == 'toy_lowrank':
        save_dir = './result_lowrank/toy_3'

        write_config_to_file(args, save_dir)
        # for t in range(args.trails):
        #     print("trial: %d" % t)
        #     fig, ax = plt.subplots()
        #     for L in batches:
        #         args.batches = int(L)
        #         if L == 10:
        #             args.c = 1.5
        #         elif L == 100:
        #             args.c = 1.01
        #         blr_rg = train_lowrank(args)
        #         torch.save(blr_rg, os.path.join(save_dir, 'blr' + str(args.d_1) + '-rg-' + str(t)))
        #         ax.plot(blr_rg, label=str(L))
        #     ax.legend()
        #     plt.show()
        #     fig.savefig(os.path.join(save_dir, 'blr' + str(args.d_1) + '-' + str(t) + '.jpg'))

        ## search parameters
        lam_set = [100]
        for lam in lam_set:
            args.lam_1 = lam
            new_dir = save_dir + '/lam' + str(lam)
            os.makedirs(new_dir, exist_ok=True)
            write_config_to_file(args, new_dir)
            for t in range(args.trails):
                print(f"trail: {t}, lam: {lam}")
                fig, ax = plt.subplots()
                blr_rg = train_LowOFUL(args)
                torch.save(blr_rg, os.path.join(new_dir, 'blroful-rg-' + str(t)))
                ax.plot(blr_rg)
                plt.show()
                fig.savefig(os.path.join(new_dir, 'blr-' + str(t) + '.jpg'))


    elif args.dataset == 'warfarin':
        save_dir = './result_warfarin'
        time_blb = []
        time_lb = []
        for t in range(args.trails):
            print("trial: %d" % t)
            start = time.time()
            blb_err_rate = train_wfr_blb(args)
            end = time.time()
            time_blb.append(start - end)

            start = time.time()
            # lb_err_rate = train_wfr_lb(args)
            end = time.time()
            time_lb.append(start - end)
            torch.save(blb_err_rate, os.path.join(save_dir, 'blb-er-' + str(t)))
            # torch.save(lb_err_rate, os.path.join(save_dir, 'lb-er-' + str(t)))
            make_figure(blb_err_rate)  # , lb_err_rate
        time_blb = np.array(time_blb).mean()
        print(time_blb)
        time_lb = np.array(time_lb).mean()
        print(time_lb)

    else:
        assert args.dataset == 'price'
        save_dir = './result_price/lam2{0}_t0{1}_q{2}_h{3}_true'.format(args.lam_2_0, args.t_0, args.q, args.h)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        write_config_to_file(args, save_dir)

        # get data
        centers = [10, 13, 41, 43, 53, 91]

        # beta_true = oracle_esti(centers, args.lam_1, device)
        beta_true = np.fromfile('beta_oracle.bin', dtype=float)
        seeds = [1, 2, 3, 4, 8, 9, 10, 11, 12, 13]

        for t in range(args.trails):
            print("trial: %d" % t)

            train_data, price_true, demand_true = data_loader(centers)
            train_data = train_data
            price_true = price_true
            demand_true = demand_true

            args.time_steps = train_data.shape[0]

            fig, ax = plt.subplots()
            blb_rg = train_price_blb(args, train_data, price_true, demand_true, beta_true)
            torch.save(blb_rg, os.path.join(save_dir, 'blb' + '-rg-' + str(t)))
            ax.plot(blb_rg, label='batched')

            lb_rg = train_price_lb(args, train_data, price_true, demand_true, beta_true)
            torch.save(lb_rg, os.path.join(save_dir, 'lb' + '-rg-' + str(t)))
            ax.plot(lb_rg, label='sequential')

            ax.legend()
            plt.show()
            fig.savefig(os.path.join(save_dir, 'blb-' + str(t) + '.jpg'))