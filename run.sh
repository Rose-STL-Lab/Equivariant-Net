#!/bin/bash
# To train models on RBC data
python3 run_model.py --architecture=ResNet --symmetry=None --output_length=3 --learning_rate=0.001 
python3 run_model.py --architecture=Unet --symmetry=None --output_length=4 --learning_rate=0.001
python3 run_model.py --architecture=ResNet --symmetry=UM --output_length=3 --learning_rate=0.001
python3 run_model.py --architecture=Unet --symmetry=UM --output_length=4 --learning_rate=0.001 
python3 run_model.py --architecture=ResNet --symmetry=Rot --output_length=3 --learning_rate=0.001
python3 run_model.py --architecture=Unet --symmetry=Rot --output_length=4 --learning_rate=0.001
python3 run_model.py --architecture=ResNet --symmetry=Mag --output_length=3 --learning_rate=0.005
python3 run_model.py --architecture=Unet --symmetry=Mag --output_length=4 --learning_rate=0.005
python3 run_model.py --architecture=ResNet --symmetry=Scale --output_length=3 --learning_rate=0.0001
python3 run_model.py --architecture=Unet --symmetry=Scale --output_length=4 --learning_rate=0.0001

