import json
from os.path import join


def split_mal_ben_client_files(files, round):
    b_files = [file for file in files if file.endswith('_b_%i.npy' % round)]
    m_files = [file for file in files if file.endswith('_m_%i.npy' % round)]

    return b_files, m_files


def get_model_name(experiment_name):
    file = join('.', 'experiments', experiment_name, 'config.json')
    with open(file) as f:
        params = json.loads(f.read())
        return params['model_name']
