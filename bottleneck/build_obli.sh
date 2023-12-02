make clean

cd ../build
cmake -DOE_DEBUG=1 -DSIMULATE=ON -DUSE_AVX2=OFF -DOBLIVIOUS=ON -DLOGGING=ON ..
make -j4

cd ../python-package
sudo python3 setup.py install

cd ../bottleneck