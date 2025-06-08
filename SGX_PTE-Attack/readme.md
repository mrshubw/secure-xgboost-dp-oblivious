## Attack Experiment

### Environment Information

- **OS**: Ubuntu 18.04, 4.15.0-175-generic #184-Ubuntu SMP  
- **CPU**: Intel Xeon Platinum 8369B @ 24x 3.522GHz  
- **SGX SDK**: Open-Enclave 0.18.5  
- **Secure XGBoost**: 0.1.1  
- **Python**: 3.8  

---

### Environment Setup

1. **Install Open-Enclave 0.18.5**  
   Follow the official guide on the Open-Enclave GitHub repository to install.

1. **Build and Install Secure XGBoost 0.1.1**  
   Follow the official guide on Secure XGBoost's GitHub to build and install.  
   To enable hardware remote attestation, set the `pccs_url` in `/etc/sgx_default_qcnl.conf` according to your platform.  
   On Ubuntu 18.04, it is recommended to use Python 3.8 or run the provided installation script:  
   ```bash
   ./secure-xgboost/make_hw.sh
   ```

1. **Apply Debug Patch**

    After Secure XGBoost is successfully installed and tested, apply the changes from `secure-xgboost/patch.diff` and rebuild the project.

    This patch adds debugging output to help simplify tracing without manual modification.

1. **Build and Install `spy-kernel` Module**

    First, install the kernel source:

    ```bash
    sudo apt-get source linux-source
    ```

    Then build and install the attack kernel module by running the following in the `spy-kernel` directory:

    ```bash
    make remake
    ```

1. **Build the `spy-user` Library**

    In the `spy-user` directory, run:

    ```bash
    make lib
    ```
    This generates the shared library `libsgx_pte_attack.so`.

### Attack Reproduction Steps
To reproduce the attack against Secure XGBoost:

1. Disable ASLR:

    ```bash
    echo 0 | sudo tee /proc/sys/kernel/randomize_va_space
    ```

1. Disable Hyper-Threading:

    ```bash
    echo off | sudo tee /sys/devices/system/cpu/smt/control
    ```
1. Run Prediction Scripts and Save Logs:

    ```bash
    python3.8 predict.py 1 > log1.log && python3.8 predict.py 2 > log2.log
    ```
1. Edit `spy-kernel/pte_utils.c`

    Set `C_ADR` to the tree_info address

    Set `A_ADR` and `B_ADR` to the addresses of `Node[254]` and `Node[374]` as they first appear in `log1.log`

1. Rebuild and Install spy-kernel:

    ```bash
    cd spy-kernel
    make remake
    ```
1. Trace Page Access Pattern for User 1:

    ```bash
    echo > /sys/kernel/debug/tracing/trace && \
    python3.8 predict.py 1 > log1.log && \
    cat /sys/kernel/debug/tracing/trace > result_user1.txt
    ```
1. Trace Page Access Pattern for User 2:

    ```bash
    echo > /sys/kernel/debug/tracing/trace && \
    python3.8 predict.py 2 > log2.log && \
    cat /sys/kernel/debug/tracing/trace > result_user2.txt
    ```
1. Compare and Analyze Results

    Analyze the two trace files to verify the attack and extract the page access patterns.

