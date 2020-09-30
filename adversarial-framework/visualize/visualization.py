import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt


def experiment(file_path, num_malicious_clients=0, attack_type=''):
    if attack_type != '':
        attack_type = '_'+attack_type
    root = os.path.join('.', 'log')
    iid_3 = pd.read_csv(os.path.join(root, '10_3_%i_IID_20_3_100%s.csv' % (num_malicious_clients, attack_type) ))['accuracy']
    iid_6 = pd.read_csv(os.path.join(root, '10_6_%i_IID_20_3_100%s.csv' % (num_malicious_clients, attack_type)))['accuracy']

    non_iid_3 = pd.read_csv(os.path.join(root, '10_3_%i_non-IID_20_3_100%s.csv' % (num_malicious_clients, attack_type)))['accuracy']
    non_iid_6 = pd.read_csv(os.path.join(root, '10_6_%i_non-IID_20_3_100%s.csv' % (num_malicious_clients, attack_type)))['accuracy']

    rounds = range(1, len(iid_3) + 1)
    fig = plt.figure()

    # IID, 3 and 6 clients
    plt.subplot(2, 1, 1)
    plt.plot(rounds, iid_3, color='r', label='3 selected clients')
    plt.plot(rounds, iid_6, color='b', label='6 selected clients')

    plt.ylim(0, 1);
    plt.xlim(1, rounds[-1])
    plt.legend()
    plt.title('IID data distribution')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.legend()

    # non-IID, 3 and 6 clients
    plt.subplot(2, 1, 2)
    plt.plot(rounds, non_iid_3, color='r', label='3 selected clients')
    plt.plot(rounds, non_iid_6, color='b', label='6 selected clients')

    plt.ylim(0, 1);
    plt.xlim(1, rounds[-1])
    plt.legend()
    plt.title('non-IID data distribution')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(file_path)


def main():
    experiment(os.path.join('.', 'figures', 'experiment_untargeted_1.pdf'), 1)
    experiment(os.path.join('.', 'figures', 'experiment_untargeted_2.pdf'), 2)

    experiment(os.path.join('.', 'figures', 'experiment_targeted3_1.pdf'), 1, 'targeted_3')
    experiment(os.path.join('.', 'figures', 'experiment_targeted3_2.pdf'), 2, 'targeted_3')

    experiment(os.path.join('.', 'figures', 'experiment_targeted3Scale2_1.pdf'), 1, 'targeted_3_True_2')
    experiment(os.path.join('.', 'figures', 'experiment_targeted3Scale2_2.pdf'), 2, 'targeted_3_True_2')


if __name__ == '__main__':
    main()
