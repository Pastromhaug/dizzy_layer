cd dizzy_layer
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
python rnn.py 6 6 0.1 1
