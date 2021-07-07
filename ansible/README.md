# fl-ansible


## Getting Started

- install pipenv `pip install pipenv`
- pipenv set up `pipenv install` in root folder
- run `pipenv shell` in root folder


### Analysis Playbook

- first run: `ansible-playbook analysis.yml -i inventory/analysis -e "exp=demo run=new"`
- continue a run (with run id): `ansible-playbook analysis.yml -i inventory/analysis -e "exp=demo run=1611332286"`

### Microbenchmark Playbook

- Start Microbenchmark and check for a fixed amount of time whether benchmark finished, then fetch results: `ansible-playbook microbench.yml -i inventory`

- Only start the microbenchmark (with specific `fp` and `frac`): `ansible-playbook microbench.yml -i inventory --tags "start" -e "fp=16 frac=8"`

- When Microbenchmark is Running, Wait until finished and then fetch results: `ansible-playbook microbench.yml -i inventory --tags "result"`

- you can add `--ssh-common-args='-o StrictHostKeyChecking=no'` as argument which means that you don't have to type `yes` when trying to connect to a newly created ec2 instance.

### E2E Playbook

#### Generating data
Open `data/preprocess.ipynb` and follow the steps to generate the data per client.
_Note: In the future, we may upload pre-processed versions to an S3 bucket for easier upload_

#### Configuration
 - Most job configuration parameters are in the .yml config files (e.g. `experiments/mnist_basic.yml`) under the `job` key.

 - Make sure the number of clients in the FL setup is divisible by the number of client machines.
If this is not the case, the client -> machine division algorithm does not work properly.
   In the future, this should be easy to fix to allow for an unbalanced division.
   
 - Other configuration parameters such as the machine type and optimization (e.g., skylake) can be found in `group_vars/all/main.yml`.

#### Running
- First run: `ansible-playbook e2ebench.yml -i inventory -e "exp=mnist_basic run=new"`
- Continue a run (with run id): `ansible-playbook e2ebench.yml -i inventory -e "exp=mnist_basic run=1611332286"`