# pylint: disable=invalid-name, exec-used
"""Setup xgboost package."""
from __future__ import absolute_import
import io
import sys
import os
from setuptools import setup, find_packages
import subprocess
sys.path.insert(0, '.')

CURRENT_DIR = os.path.dirname(__file__)

# We can not import `xgboost.libpath` in setup.py directly since xgboost/__init__.py
# import `xgboost.core` and finally will import `numpy` and `scipy` which are setup
# `install_requires`. That's why we're using `exec` here.
libpath_py = os.path.join(CURRENT_DIR, 'securexgboost/libpath.py')
libpath = {'__file__': libpath_py}
exec(compile(open(libpath_py, "rb").read(), libpath_py, 'exec'), libpath, libpath)

LIB_PATH = []
for libfile in libpath['find_lib_path']():
    try:
        relpath = os.path.relpath(libfile, CURRENT_DIR)
        LIB_PATH.append(relpath)
        break  # need only one
    except ValueError:
        continue

print("Updating protos")
subprocess.run(["python3", "-m", "grpc_tools.protoc", "-I", "securexgboost/rpc/protos", "--python_out=securexgboost/rpc", "--grpc_python_out=securexgboost/rpc", "securexgboost/rpc/protos/remote.proto", "securexgboost/rpc/protos/ndarray.proto"])

print("Install libxgboost from: %s" % LIB_PATH)
# Please use setup_pip.py for generating and deploying pip installation
# detailed instruction in setup_pip.py
setup(name='securexgboost',
      version=open(os.path.join(CURRENT_DIR, 'securexgboost/VERSION')).read().strip(),
      description="Secure XGBoost Python Package",
      long_description=io.open(os.path.join(CURRENT_DIR, 'README.rst'), encoding='utf-8').read(),
      install_requires=[
          'numpy',
          'scipy',
      ],
      maintainer='Hyunsu Cho',
      maintainer_email='chohyu01@cs.washington.edu',
      zip_safe=False,
      packages=find_packages(),
      # this will use MANIFEST.in during install where we specify additional files,
      # this is the golden line
      include_package_data=True,
      data_files=[('securexgboost', LIB_PATH)],
      license='Apache-2.0',
      classifiers=['License :: OSI Approved :: Apache Software License',
                   'Development Status :: 5 - Production/Stable',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7'],
      python_requires='>=3.4',
      url='https://github.com/dmlc/xgboost')

