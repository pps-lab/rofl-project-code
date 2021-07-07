
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import yaml
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
EXPERIMENT_RESULTS_PATH = "/Users/hidde/PycharmProjects/fl-ansible/experiment_results"

def build_femnist_bounds_plot(folder='femnist_outliers_bound_nopgd_1613384798'):

    base_path = os.path.join(EXPERIMENT_RESULTS_PATH, folder)

    num_runs = 18

    data_metrics = [pd.read_csv(os.path.join(base_path, f"run_{i}", "log.csv")) for i in range(num_runs)]
    data_configs = [yaml.load(open(os.path.join(base_path, f"run_{i}", "config.yml")), Loader=yaml.FullLoader) \
                    for i in range(num_runs)]

    num_last = 100
    offset_last = data_metrics[0]['round'].max() - num_last
    print(offset_last)
    # print(data_metrics)
    # print(data_configs)


    # print(m[(m['round'] == 1000)]['adv_success'])
    # print([m[m['round'] == 1000] for m in data_metrics])

    data_bounds = np.array([c['hyperparameters']['args']['client']['clip']['value'] for c in data_configs])
    data_adv = np.array([m[m['round'] > offset_last]['adv_success'].mean() for m in data_metrics])

    print(data_metrics[0][data_metrics[0]['round'] > offset_last]['adv_success'])
    print(data_adv)

    # sorted = np.argsort(data_bounds)
    sorted = np.concatenate([[3, 4, 5, 0, 1, 2], list(range(6, num_runs))])
    data_bounds = data_bounds[sorted]
    data_adv = data_adv[sorted]

    idx_prot = range(0, data_bounds.shape[0], 3)
    idx_outlier = range(1, data_bounds.shape[0], 3)
    idx_rand = range(2, data_bounds.shape[0], 3)
    print(list(idx_prot), list(idx_outlier), list(idx_rand))

    # data_adv[idx_outlier] += 0.35

    plt.figure()
    # plt.scatter([c['hyperparameters']['args']['client']['clip']['value'] for c in data_configs],
    #             [m[m['round'] == 1000]['adv_success'] for m in data_metrics])
    plt.plot(data_bounds[idx_outlier], data_adv[idx_outlier], '-ro', label='Outliers')
    plt.plot(data_bounds[idx_rand], data_adv[idx_rand], '-x', label='Random')
    plt.plot(data_bounds[idx_prot], data_adv[idx_prot], '-o', label='Prototypes')
    plt.ylabel('Adversarial success')
    plt.xlabel('Norm bound ($L_2$)')
    plt.legend()
    plt.show()


def build_femnist_bounds_merge_two_plot(folder_one='femnist_outliers_bound_nopgd_1613384798',
                                        # folder_two='femnist_outliers_bound_nopgd_1613401227'):
                                        folder_two='femnist_outliers_bound_nopgd_1613900998'):
    def get_metr(folder, runs_to_include):
        base_path = os.path.join(EXPERIMENT_RESULTS_PATH, folder)

        data_metrics = [pd.read_csv(os.path.join(base_path, f"run_{i}", "log.csv")) for i in runs_to_include]
        data_configs = [yaml.load(open(os.path.join(base_path, f"run_{i}", "config.yml")), Loader=yaml.FullLoader) \
                        for i in runs_to_include]
        return data_metrics, data_configs

    met_one, conf_one = get_metr(folder_one, range(12))
    met_two, conf_two = get_metr(folder_two, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17])

    data_metrics = met_one + met_two
    data_configs = conf_one + conf_two

    num_last = 100
    offset_last = data_metrics[0]['round'].max() - num_last
    print(offset_last)
    # print(data_metrics)
    # print(data_configs)


    # print(m[(m['round'] == 1000)]['adv_success'])
    # print([m[m['round'] == 1000] for m in data_metrics])

    data_bounds = np.array([c['hyperparameters']['args']['client']['clip']['value'] for c in data_configs])
    data_type = np.array([c['hyperparameters']['args']['environment']['malicious_client_indices'][0] for c in data_configs])
    data_adv = np.array([m[m['round'] > offset_last]['adv_success'].mean() for m in data_metrics])
    data_err = np.array([m[m['round'] > offset_last]['adv_success'].std() for m in data_metrics])

    print(data_metrics[0][data_metrics[0]['round'] > offset_last]['adv_success'])
    print(data_type)
    print(data_adv)
    print(data_err)

    data_by_type = []
    for elem in set(data_type):
        this_adv = data_adv[data_type == elem]
        this_bounds = data_bounds[data_type == elem]
        this_err = data_err[data_type == elem]
        sorted = np.argsort(this_bounds)
        data_by_type.append((this_bounds[sorted], this_adv[sorted], this_err[sorted]))

    # data_adv[idx_outlier] += 0.35

    # plt.scatter([c['hyperparameters']['args']['client']['clip']['value'] for c in data_configs],
    #             [m[m['round'] == 1000]['adv_success'] for m in data_metrics])
    plt.figure()
    plt.errorbar(data_by_type[0][0], data_by_type[0][1], fmt='-ro', label='Random', yerr=data_by_type[0][2], elinewidth=3)
    plt.errorbar(data_by_type[1][0], data_by_type[1][1], fmt='-x', label='Outliers', yerr=data_by_type[1][2], elinewidth=3)
    plt.errorbar(data_by_type[2][0], data_by_type[2][1], fmt='-o', label='Prototypes', yerr=data_by_type[2][2], elinewidth=3)
    plt.hlines(1.0, 1, 10, linestyles='dashed')
    plt.ylabel('Adversarial success')
    plt.xlabel('Norm bound ($L_2$)')
    plt.legend()
    plt.show()

def build_femnist_success_plot(folder='femnist_outliers_spectrum_1612963984'):

    base_path = os.path.join(EXPERIMENT_RESULTS_PATH, folder)

    data_metrics = [pd.read_csv(os.path.join(base_path, f"run_{i}", "log.csv")) for i in range(10)]
    data_configs = [yaml.load(open(os.path.join(base_path, f"run_{i}", "config.yml")), Loader=yaml.FullLoader) for i in range(10)]

    # data_bounds = np.array([c['hyperparameters']['args']['client']['clip']['value'] for c in data_configs])
    data_bounds = list(range(len(data_metrics)))
    data_adv = np.array([m[m['round'] > 900]['adv_success'].mean() for m in data_metrics])

    print(data_metrics[0][data_metrics[0]['round'] > 900]['adv_success'])
    print(data_adv)

    # sorted = np.argsort(data_bounds)
    # data_bounds = data_bounds[sorted]
    # data_adv = data_adv[sorted]

    idx_prot = range(0, len(data_bounds), 3)
    idx_outlier = range(1, len(data_bounds), 3)
    idx_rand = range(2, len(data_bounds), 3)
    print(list(idx_prot), list(idx_outlier), list(idx_rand))

    # data_adv[idx_outlier] += 0.35

    plt.figure()
    plt.plot(data_bounds, data_adv)
    plt.ylabel('Adversarial success')
    plt.xlabel('Outlierness')
    plt.legend()
    plt.show()


def build_grid_search_plot(folder='cifar_single_shot_1615458133'):

    base_path = os.path.join(EXPERIMENT_RESULTS_PATH, folder)

    num_runs = 40

    data_metrics = [pd.read_csv(os.path.join(base_path, f"run_{i}", "log.csv")) for i in range(num_runs)]
    data_configs = [yaml.load(open(os.path.join(base_path, f"run_{i}", "config.yml")), Loader=yaml.FullLoader) \
                    for i in range(num_runs)]

    print(data_metrics)

    data_epochs = np.array([c['hyperparameters']['args']['client']['malicious']['objective']['args']['num_epochs'] for c in data_configs])
    data_poison_samples = np.array([c['hyperparameters']['args']['client']['malicious']['objective']['args']['poison_samples'] for c in data_configs])
    data_adv = np.array([m[m['round'] == 1]['adv_success'][0] for m in data_metrics])
    data_test = np.array([m[m['round'] == 1]['accuracy'][0] for m in data_metrics])

    plt.figure()
    plt.hexbin(data_epochs, data_poison_samples, C=data_test, gridsize=30, cmap=cm.jet, bins=None)
    plt.axis([data_epochs.min(), data_epochs.max(), data_poison_samples.min(), data_poison_samples.max()])
    plt.colorbar()
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(data_epochs, data_poison_samples, data_adv, c='red')
    # ax.scatter(data_epochs, data_poison_samples, data_test, c='blue')
    #
    # plt.show()

    print(data_epochs)
    print(data_adv)

    # sorted = np.argsort(data_bounds)
    sorted = np.concatenate([[3, 4, 5, 0, 1, 2], list(range(6, num_runs))])
    data_bounds = data_bounds[sorted]
    data_adv = data_adv[sorted]

    idx_prot = range(0, data_bounds.shape[0], 3)
    idx_outlier = range(1, data_bounds.shape[0], 3)
    idx_rand = range(2, data_bounds.shape[0], 3)
    print(list(idx_prot), list(idx_outlier), list(idx_rand))

    # data_adv[idx_outlier] += 0.35

    plt.figure()
    # plt.scatter([c['hyperparameters']['args']['client']['clip']['value'] for c in data_configs],
    #             [m[m['round'] == 1000]['adv_success'] for m in data_metrics])
    plt.plot(data_bounds[idx_outlier], data_adv[idx_outlier], '-ro', label='Outliers')
    plt.plot(data_bounds[idx_rand], data_adv[idx_rand], '-x', label='Random')
    plt.plot(data_bounds[idx_prot], data_adv[idx_prot], '-o', label='Prototypes')
    plt.ylabel('Adversarial success')
    plt.xlabel('Norm bound ($L_2$)')
    plt.legend()
    plt.show()

# build_femnist_bounds_plot()
# build_femnist_bounds_merge_two_plot()
# build_femnist_success_plot()
build_grid_search_plot()