#!/bin/bash

set -ex

# make libxgboost.so
source /opt/openenclave/share/openenclave/openenclaverc
cd ../..
rm -rf build/
mkdir build
cd build
# hardware+release
cmake -DSIMULATE=OFF -DLOGGING=ON ..
# hardware+debug
# cmake -DSIMULATE=OFF -DOE_DEBUG=1 -DLOGGING=ON ..
make -j32

# makke python package
cd ../python-package/
python3.8 -m grpc_tools.protoc -I securexgboost/rpc/protos --python_out=securexgboost/rpc --grpc_python_out=securexgboost/rpc securexgboost/rpc/protos/remote.proto securexgboost/rpc/protos/ndarray.proto
sudo python3.8 setup.py install