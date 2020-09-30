import os

import boto3

BUCKET_NAME = 'attacks-on-federated-learning'
EXPERIMENT_DIR = 'experiments'


def fetch_files():
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=BUCKET_NAME)
    return response['Contents']


def download_file(key):
    print(key['Key'])
    exp_dir, exp_name, file = key['Key'].split(os.path.sep)
    save_dir = os.path.join(exp_dir, exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    s3 = boto3.resource('s3')
    s3.Bucket(BUCKET_NAME).download_file(key['Key'], os.path.join(save_dir, file))


def main():
    bucket_files = fetch_files()
    for file in bucket_files:
        if file['Key'].startswith(EXPERIMENT_DIR):
            if file['Key'].endswith('log.csv') or file['Key'].endswith('config.json'):
                download_file(file)


if __name__ == '__main__':
    main()
