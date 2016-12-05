export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
python runFilterProblem.py blah 6 6 0.1 1
