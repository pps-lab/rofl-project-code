from baseline import get_df, print_table, make_plots


def main():
    for attack, objective in [('untargeted', 'adv_success'), ('targeted', 'adv_success')]:
        for scale_attack in [True, False]:
            if attack == 'untargeted':
                for untargeted_after in [True, False]:
                    wildcard = f'{attack}_mal_baseline*'
                    if untargeted_after:
                        wildcard = f'{attack}_true_mal_baseline*'
                    df = get_df(wildcard)
                    df = df[df.scale_attack == scale_attack]
                    for dataset in ['mnist', 'fmnist']:
                        print('\n', attack, objective, dataset, f'scale_attack={scale_attack}', ':')
                        print_table(df, dataset, -1, objective=objective)

                        file_prefix = f'{attack}_true_mal_baseline' if untargeted_after else f'{attack}_mal_baseline'
                        if scale_attack:
                            file_prefix += '_boosting'

                        # todo fix min of y_axis
                        make_plots(df, dataset, -1,
                                   attack_type=attack,
                                   file_prefix=file_prefix,
                                   plot_dir=f'mal_baseline_plots', objective=objective,
                                   ylabel='Adversarial success')
            else:
                wildcard = f'{attack}_mal_baseline*'
                df = get_df(wildcard)
                df = df[df.scale_attack == scale_attack]
                for dataset in ['mnist', 'fmnist']:
                    print('\n', attack, objective, dataset, f'scale_attack={scale_attack}', ':')
                    print_table(df, dataset, -1, objective=objective)

                    file_prefix = f'{attack}_mal_baseline_boosting' if scale_attack else f'{attack}_mal_baseline'
                    # todo fix min of y_axis
                    make_plots(df, dataset, -1,
                               attack_type=attack,
                               file_prefix=file_prefix,
                               plot_dir=f'mal_baseline_plots', objective=objective,
                               ylabel='Adversarial success')


if __name__ == '__main__':
    main()
