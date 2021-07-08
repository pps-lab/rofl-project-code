<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://pps-lab.com/research/ml-sec/">
    <img src="https://github.com/pps-lab/fl-analysis/blob/master/documentation/ml-sec-square.png?raw=true" alt="Logo" width="80" height="80">  
  </a>

<h2 align="center"><u>RoFL</u>: Attestable <u>Ro</u>bustness for <u>F</u>ederated <u>L</u>earning</h2>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open"> 
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#requirements">Requirements</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
        <a href="#usage">Usage</a>
        <ul>
        <li><a href="#using-ansible">Using Ansible</a></li>
        <li><a href="#manually">Manually</a></li>
      </ul>
    </li>
    <li><a href="#logging">Logging</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About the Project
This framework is an end-to-end implementation of the protocol proposed in 
of [RoFL: Attestable Robustness for Secure Federated Learning](https://arxiv.org/abs/2107.03311).
The protocol relies on combining secure aggregation with commitments together with
zero-knowledge proofs to proof constraints on client updates.

The current implementation of RoFL is an academic proof-of-concept prototype. 
The prototype is designed to focus on evaluating the overheads of zero-knowledge proofs on client updates on top of secure aggregation. 
The current prototype is not meant to be directly used for applications in productions. 

This repository is structured as follows.
### RoFL components
* [RoFL_Service](): This directory contains the code for the federated learning server and client, written in Rust.
* [RoFL_Crypto](): This directory contains the cryptographic library to generate and verify the zero-knowledge proof constraints
used in RoFL.
* [RoFL_Train_Client](): This directory contains a python service to handle the training and inference operations for the machine learning model.
The RoFL_Service interfaces with the RoFL_Train_Client using gRPC.
This component acts as a wrapper around the [FL Analysis framework](https://github.com/pps-lab/fl-analysis) that is used for machine learning training.

### Utilities
* [ansible](): The Ansible setup used for the evaluation of the framework. 
  For more information on how to use this, See [ansible/README.md](https://github.com/pps-lab/rofl-project-code/blob/master/ansible/README.md).

* [plots](): This directory contains code used to generate plots for the paper.


### End-to-end implementation
The end-to-end setup consists of two components.
First the secure federated learning with constraints implementation performs the communication between
the server and the clients.
On the client- and server-side, this component offloads the machine learning operations to a python training and evaluation service.


<!-- GETTING STARTED -->
## Getting Started

Follow these steps to run the implementation on your local machine.

### Requirements
Python 3.7 & Rust version:
```
Min: 1.52.0-nightly 
```

### Installation
Both the secure FL component and the training service are installed separately.

#### Secure FL with constraints
1. Clone this repository
```sh
git clone git@github.com:pps-lab/rofl-project-code.git
```
2. Install Cargo/Rust
```
curl https://sh.rustup.rs -sSf | sh -s -- -y
```
3. Switch to nightly

_Note:_ As of now, only a specific nightly version is supported due to a deprecated feature that a dependency is using.   
`rustup override set nightly-2021-05-11`
4. `cargo build`
   
#### Python training service
1. Install the requirements for the trainservice wrapper (in `rofl_project_code`)
```
cd rofl_train_client
pip install -r requirements.txt
```
2. Download the analysis framework
```
cd ../../ # go up to workspace directory
git clone git@github.com:pps-lab/fl-analysis.git
```
3. Install the requirements for the analysis framework
```sh
cd fl-analysis
pipenv install
```

## Usage
The framework can be used in two ways.

### Using Ansible
We provide a setup in Ansible to easily deploy and evaluate the framework on multiple servers on AWS.
See [ansible/README.md](https://github.com/pps-lab/rofl-project-code/blob/master/ansible/README.md) for instructions on how to use this Ansible setup.

### Manually
To run the setup manually, several components must be run separately.
The individual components must be started in this order.
The example shown are for a basic local configuration with four clients with L8-norm (infinity) range proof verification.
The implementation of RoFL uses the analysis framework for model training and evaluation.
In the following, we assume the following directory structure:

Top-level directory (e.g., workspace):
- `rofl-project-code` (this repository)
- `fl-analysis` (the analysis framework)

Each component must be run in a separate terminal window.

#### Server
In `rofl-project-code`, run the server:
```
./target/debug/flserver
```

#### Client Trainer
First, navigate to the analysis framework directory and enter the pipenv:
```sh
cd ../fl-analysis
pipenv shell
```
Then, navigate back to the python directory in the implementation directory:
```sh
cd ../rofl-project-code/rofl_train_client
```
From the `rofl_train_client` directory, run the python service.
```
cd rofl_train_client
PYTHONPATH=$(pwd) python trainservice/service.py
```

#### Client
In the `rofl-project-code` directory, run the client executable.
```
cd ../
./target/debug/flclients -n 4 -r 50016
```

#### Observer (optional)
After running the client, training has started.
In addition, the observer component can be used to evaluate the model accuracy on the server-side.
To do so, first, navigate to the analysis framework directory and enter the pipenv:
```sh
cd ../fl-analysis
pipenv shell
```
Then, navigate back to the python directory in the implementation directory:
```sh
cd ../rofl-project-code/rofl_train_client
```
Set the PYTHONPATH to include the current directory and run
```
PYTHONPATH=$(pwd) python trainservice/observer.py
```
The observer will connect to the FL server and receive the global model for each round.

## Logging
The implementation outputs time and bandwidth measurements in several files.

### Benchmark Log Format
The benchmark files for both the server and the clients can be found in the benchlog folder.

#### Format of the server log
```
t1--t2--t3--t4
t1: round starts
t2: round aggregation done
t3: round param extraction done
t4: verification completes

Format of a benchmark log line:
<Round ID>, <t2 - t1>, <t3 - t2>, <t4 - t3>, <total duration>
```

#### Format of the client log
```
t1--t2--t3--t4--t5
t1: model meta received
t2: model completely received
t3: local model training done
t4: model update encryption + proofs completed
t5: model sent to server

Format of a benchmark log line:
<Round ID>, <t2 - t1>, <t3 - t2>, <t4 - t3>, <t5 - t4>, <total duration>, <bytes received>,  <bytes sent>
```

<!-- LICENSE -->
## License

This project's code is distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

* Hidde Lycklama - [hiddely](https://github.com/hiddely)
* Lukas Burkhalter - [lubux](https://github.com/lubux)

Project Links: [https://github.com/pps-lab/rofl-project-code](https://github.com/pps-lab/fl-project-code) and [https://pps-lab.com/research/ml-sec/](https://pps-lab.com/research/ml-sec/)
