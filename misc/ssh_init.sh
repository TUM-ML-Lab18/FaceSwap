#!/bin/sh
ssh -N -f -L localhost:2828:localhost:2828 gpu05 # JupyterLab
ssh -N -f -L localhost:7007:localhost:7007 gpu05 # TensorBoard
ssh -N -f -L localhost:6666:localhost:22 gpu05 # PyCharm
ssh -N -f -L localhost:6006:localhost:6006 gpu05 # TensorBoard Sebastian
