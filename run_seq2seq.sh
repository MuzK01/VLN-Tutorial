#!/bin/bash
# Wrapper script to run Python with the correct cuDNN version

# Save the original LD_LIBRARY_PATH
ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# Unset LD_LIBRARY_PATH to use PyTorch's bundled cuDNN
export LD_LIBRARY_PATH=""

# Run the Python script with all arguments passed to this script
python seq2seq/train.py

# Restore the original LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH 