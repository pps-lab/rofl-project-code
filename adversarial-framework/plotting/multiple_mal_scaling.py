from os.path import join

from baseline import make_plots, get_df


def main():
    dataset = 'mnist'
    for dataset in ['mnist', 'fmnist']:
        for i in [2., 3., 5.]:
            # scale plots
            df = get_df('scale_mal_*')
            df_copy = df[df.scale_attack_weight == i].copy()
            for attack_type in ['untargeted', 'targeted']:
                make_plots(df_copy, dataset, -1, attack_type, 'scale_%i_%s' % (i, attack_type),
                           join(scale_plots_dir, dataset), ylabel, objective)

            # multi_mal plots
            df = get_df('multiple_mal_*')
            df_copy = df[df.num_malicious_clients == int(i)].copy()
            for attack_type in ['untargeted', 'targeted']:
                make_plots(df_copy, dataset, -1, attack_type, 'multiple_mal_%i_%s' % (i, attack_type),
                           join(multiple_mal_plots_dir, dataset),  ylabel, objective)


if __name__ == '__main__':
    scale_plots_dir = 'scale_plots'
    multiple_mal_plots_dir = 'multiple_mal_plots'
    ylabel = 'Adversarial success'
    objective = 'adv_success'
    main()
