#!/bin/bash

python3 /mnt/code/benchmarks/main3d_singletask.py --dataname BDE --target BindingEnergy --modeltype ChIRo
python3 /mnt/code/benchmarks/main3d_singletask.py --dataname BDE --target BindingEnergy --modeltype ChIRo
python3 /mnt/code/benchmarks/main3d_singletask.py --dataname Drugs --target ip --modeltype ChIRo
python3 /mnt/code/benchmarks/main3d_singletask.py --dataname Drugs --target ea --modeltype ChIRo
python3 /mnt/code/benchmarks/main3d_singletask.py --dataname Drugs --target chi --modeltype ChIRo
python3 /mnt/code/benchmarks/main3d_singletask.py --dataname Kraken --target sterimol_B5 --modeltype ChIRo
python3 /mnt/code/benchmarks/main3d_singletask.py --dataname Kraken --target sterimol_L --modeltype ChIRo
python3 /mnt/code/benchmarks/main3d_singletask.py --dataname Kraken --target sterimol_burB5 --modeltype ChIRo
python3 /mnt/code/benchmarks/main3d_singletask.py --dataname Kraken --target sterimol_burL --modeltype ChIRo