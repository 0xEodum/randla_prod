# Windows users: run compile_wrappers.bat (calls compile_wrappers.ps1)
#!/bin/bash

# Compile cpp subsampling
cd cpp_subsampling
python3 setup.py build_ext --inplace
cd ..

