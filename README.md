# DOXIE

DOXIE is a secure XGBoost inference system that strikes a balance between access pattern protection and concrete efficiency. DOXIE satisfies a security definition called differentially oblivious (DO) that originates from the celebrated notion of differential privacy. We implemented the DOXIE algorithm based on the Secure XGBoost code.

DOXIE's main test code is located in the `./do-enhanced` folder.

## Setting Environment
* Installation.

    Refer to README.sxgboost.md

* Install python dependencies. Installing python dependencies as described in the installation section may fail, in which case you can try the following:

    ```sh
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Preprecessing
* Build and install the python package of XGBoost with DOXIE. You can compile and install the oblivious version of the XGBoost package using the `--O` parameter, and the non-oblivious version without any parameters.

    ```sh
    cd do-enhanced
    ./build_project.sh --DO
    ```

* Process the dataset if needed. In `do-enhanced/data/higgs/`, `data1000.enc, data10000.enc, data100000.enc` have been avaliable for higgs dataset with 1k, 10k, 100k records respectively. To use higgs dataset with other sizes, download the dataset from [HIGGS](https://archive.ics.uci.edu/dataset/280/higgs), then run

    ```sh
    cd do-enhanced
    python3 process_dataset.py --dataset higgs --size 100000
    ```
    
* If you want to test models of different sizes on other datasets, first modify the configuration in `train.py` as needed, then run the following command to train your desired model.

    ```sh
    cd do-enhanced
    python3 train.py
    ```

## Running Experiments
* In the `do-enhanced` folder, the `run_experiment.py` script runs the Non-Oblivious, DOXIE, and Oblivious inference algorithms under various experimental settings, including dataset, data size, number of trees, tree depth, privacy budget, and shuffle methods. Each configuration is run 10 times to ensure result stability. It took us two days to complete all experiments and collect the results.

* You can modify the functions called within the `run_experiment.py` script to run different experiments. For example, `experiment_throughput` measures the latency and throughput of the DOXIE algorithm; `experiment_random_forest` evaluates DOXIE’s performance when using random forests; and `experiment_shufflers` analyzes the impact of different shuffling methods on DOXIE’s performance.

* Handle the experiment log to get the test result after running the test scripts.

    ```sh
    cd do-enhanced
    python3 handle_log.py
    ```

## Attack Experiments
See `SGX_PTE-attck`.