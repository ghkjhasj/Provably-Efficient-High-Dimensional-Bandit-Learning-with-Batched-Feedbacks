import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_curve_data(model='blb', folder_path='./result', curve_type=None):
    filenames = [name for name in os.listdir(folder_path) if name.startswith(model + '-' + curve_type)]
    paths = [os.path.join(folder_path, name) for name in filenames]
    keys = [name.split('-')[2] for name in filenames]
    return {key: torch.load(fp, map_location=torch.device('cpu')) for key, fp in zip(keys, paths)}


def make_plot(curvetype, models, folder_path):
    curve_models_data = {model: get_curve_data(model, folder_path, curvetype) for model in models}
    fig, ax = plt.subplots()
    for model in models:
        curve_data = curve_models_data[model]
        print(model)
        print(curve_data)
        datas = []
        trails = 5
        for t in range(trails):
            data = curve_data[str(t)]
            datas.append(np.array(data).tolist())
        datas = np.array(datas)
        data_mean = np.mean(datas, axis=0)
        ci = 1.96 * np.std(datas, axis=0) / np.sqrt(np.size(datas, 0))
        if model == 'blr10':
            label_plot = 'Batched Bandit (L=10)'
        elif model == 'blr100':
            label_plot = 'Batched Bandit (L=100)'
        elif model == 'blr10000':
            label_plot = 'Batched Bandit (L=T)'
        elif model == 'blroful':
            label_plot = 'OFUL'
        elif model == 'blb400':
            label_plot = 'Batched Bandit (d=400)'
        elif model == 'blb800':
            label_plot = 'Batched Bandit (d=800)'
        elif model == 'blb1000':
            label_plot = 'Batched Bandit (d=1000)'
        elif model == 'blb':
            label_plot = 'Adaptive-Batched Bandit'
        elif model == 'blbf':
            label_plot = 'Fixed-Batched Bandit'
        elif model == 'blb100':
            label_plot = 'Batched Bandit (L=100)'
        elif model == 'blb10':
            label_plot = 'Batched Bandit (L=10)'
        else:
            label_plot = 'LASSO Bandit'
        T = data.shape[0]
        axis = np.arange(1, T+1)
        axis = np.log(axis) ** 2
        plt.plot(axis, data_mean, label=label_plot, linewidth=2)
        plt.fill_between(axis, (data_mean - ci), (data_mean + ci), alpha=0.1)
    #plt.xlabel("Number of Patients")
    #plt.ylabel("Fraction of Incorrect Decisions")
    plt.xlabel("$\log^2 t$")
    plt.ylabel("Cumulative Regret")
    plt.grid(linestyle='--', linewidth=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlim((0, 85))
    plt.ylim((0, 1500))
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'reg_price.pdf'))
    plt.show()
    plt.close()


def make_plot_d(curvetype, models, folder_path):
    curve_models_data = {model: get_curve_data(model, folder_path, curvetype) for model in models}
    fig, ax = plt.subplots()
    axis = np.array([10, 20, 30, 40, 50])
    axis = np.log(axis ** 2)  #+ (np.log(10000) ** 2) * np.log(axis ** 2)
    data_means = []
    cis = []
    for model in models:
        curve_data = curve_models_data[model]
        datas = []
        trails = 5
        for t in range(trails):
            data = curve_data[str(t)][9999]
            datas.append(data)
        datas = np.array(datas)
        print(datas)
        data_mean = np.mean(datas, axis=0)
        ci = 1.96 * np.std(datas, axis=0) / np.sqrt(np.size(datas, 0))
        data_means.append(data_mean)
        cis.append(ci)
    data_means = np.array(data_means)
    cis = np.array(cis)
    plt.plot(axis, data_means, linewidth=2)
    plt.fill_between(axis, (data_means - cis), (data_means + cis), alpha=0.1)
    plt.grid(linestyle='--', linewidth=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlim((4, 8))
    plt.ylim((250, 1000))
    plt.xlabel("$\log(d_1 d_2)$")
    plt.ylabel("Cumulative Regret")
    plt.savefig('./result_lowrank/reg.pdf')
    plt.show()
    plt.close()


if __name__ == '__main__':
    labels_lr = ['blr10', 'blr100', 'blr10000', 'blroful']  # 'blr10', 'blr100', 'blr10000', 'blb10', 'blb100', 'lb', 'blb200', 'blb400', 'blb800', "blb1000"; 'blb10', 'blb100'
    labels_d = ['blr10', 'blr20', 'blr30', 'blr40', 'blr50']  # 'blb100', 'blb200', 'blb400', 'blb600', 'blb800', "blb1000"
    labels_price = ['lb', 'blb']
    #make_plot('rg', labels_price, './result_price/lam21e-05_t010_q1_h5.0_true')
    make_plot('rg', labels_lr, './result_lowrank/toy_2')
    #make_plot_d('rg', labels_d, './result_lowrank/toy_d')