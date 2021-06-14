### Rust Compiler Requirements

```
Min: 1.52.0-nightly 
```

### Running the setup

#### Setting up
1. Install Cargo
1. Switch to nightly
1. `cargo build`
1. Pip install trainservice
1. Pip install analysis framework

Server: `./target/debug/flserver`
Observer: `PYTHONPATH=$(pwd) python trainservice/observer.py`
Client Trainer: 
```
Server: ./target/debug/flserver

```

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
