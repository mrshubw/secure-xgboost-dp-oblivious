#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Define variables
VENV_DIR="$HOME/secure-xgboost-dp-oblivious/.venv"
REPO_URL="https://github.com/mc2-project/secure-xgboost.git"
BUILD_DIR="$HOME/secure-xgboost-dp-oblivious/build"
PYTHON_VERSION="python3"

# Function to create virtual environment
create_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        $PYTHON_VERSION -m venv $VENV_DIR
    fi
}

# Function to activate virtual environment
activate_venv() {
    source $VENV_DIR/bin/activate
}

# Function to install necessary Python packages
install_python_dependencies() {
    pip install --upgrade pip
    pip install numpy pandas sklearn numproto grpcio grpcio-tools requests
}

# Function to install system dependencies
install_system_dependencies() {
    sudo apt -y install clang-8 libssl-dev gdb libsgx-enclave-common libsgx-quote-ex libprotobuf10 libsgx-dcap-ql libsgx-dcap-ql-dev az-dcap-client open-enclave=0.17.1
    source /opt/openenclave/share/openenclave/openenclaverc

    wget https://github.com/Kitware/CMake/releases/download/v3.15.6/cmake-3.15.6-Linux-x86_64.sh
    sudo bash cmake-3.15.6-Linux-x86_64.sh --skip-license --prefix=/usr/local

    sudo apt-get install -y libmbedtls-dev
}

# Function to clone the repository
clone_repository() {
    if [ ! -d "$HOME/secure-xgboost" ]; then
        git clone $REPO_URL $HOME/secure-xgboost
    fi
}

# Function to build the project with specified parameters
build_project() {
    # mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    cmake "$@" ..
    make -j4
}

# Function to install the Python package
install_python_package() {
  cd $HOME/secure-xgboost-dp-oblivious/python-package
  python3 setup.py install
  cd ../do-enhanced
}


# Parse input arguments to determine build configuration
BUILD_TYPE="default"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --O) BUILD_TYPE="obli" ;;
        --DO) BUILD_TYPE="DPObli" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Execute functions
activate_venv

# Determine the cmake parameters based on the build type
case $BUILD_TYPE in
    obli)
        build_project -DOE_DEBUG=1 -DSIMULATE=OFF -DUSE_AVX2=OFF -DOBLIVIOUS=ON -DDPOBLIVIOUS=OFF -DLOGGING=ON
        ;;
    DPObli)
        build_project -DOE_DEBUG=1 -DSIMULATE=OFF -DUSE_AVX2=OFF -DOBLIVIOUS=ON -DDPOBLIVIOUS=ON -DLOGGING=ON
        ;;
    default)
        build_project -DOE_DEBUG=1 -DSIMULATE=OFF -DUSE_AVX2=OFF -DOBLIVIOUS=OFF -DDPOBLIVIOUS=OFF -DLOGGING=ON
        ;;
    *)
        echo "Unknown build type: $BUILD_TYPE"
        exit 1
        ;;
esac

install_python_package

# Deactivate the virtual environment (if desired)
deactivate

# Print message to indicate the build is complete
echo "Build completed."
echo "Build type: $BUILD_TYPE"
echo "Virtual environment deactivated."