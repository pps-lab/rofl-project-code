# Federated Learning

## Setup

Only tested on Ubuntu 18.04

Please make sure you have pip3 and rustup (on nightly) installed, otherwise run:

```
    sudo apt install python3-pip
    curl https://sh.rustup.rs -sSf | sh
    rustup override set nightly 
```
Install the required packages with:

```
    cd /path/to/fed_learning
    pip install -r requirements.txt
```

The rust-packages are downloaded and installed once upon first compilation

## Run Test

Run with development configurations (lower sample size, fewer clients, smaller ML model) (needs gnome frontend):

```
    cd /path/to/fed_learning
    python scripts/start_test.py config/dev_config.ini
```

In order to modify the setup just change the path to the .ini file to your own.

## Cryptographic library

To recompile the rust library with a certain fixed precision size and fractional part run:

```
    cd /path/to/fed_learning
    cd fed_learning/crypto/crypto_interface/rust_crypto
    cargo build --release --features "fp<fp_bits> frac<frac_bits>"
```

where <fp_bits> is the number of bits for the fixed precision representation (8, 16, 32) and <frac_bits> the number of fractional bits (0 to 12)

IMPORTANT: make sure <fp_bits> and <frac_bits> match with the .ini file! Otherwise the aggregation protocol assumes the wrong fixed precision representation!
