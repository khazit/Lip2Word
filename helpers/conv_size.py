"""
Given an input size and some parameters, compute the output size of a convolution
"""

import sys

if (sys.argv[1] == "h") :
    print("python3 conv_size.py [input_size] [window_size] [stride] [padding_type]")
else :
    input_size = int(sys.argv[1])
    window_size = int(sys.argv[2])
    stride = int(sys.argv[3])
    padding_type = sys.argv[4]
    if padding_type == "valid" :
        output_size = round( (input_size-window_size) / stride) + 1
    print("Output size : {}".format(output_size))
