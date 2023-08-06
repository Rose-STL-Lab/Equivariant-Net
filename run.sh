#!/bin/bash
# RBC Data
python3 run_model.py --dataset=RBC --architecture=ResNet --symmetry=None --output_length=3 --learning_rate=0.001 
python3 run_model.py --dataset=RBC --architecture=Unet --symmetry=None --output_length=4 --learning_rate=0.001
python3 run_model.py --dataset=RBC --architecture=ResNet --symmetry=UM --output_length=3 --learning_rate=0.001
python3 run_model.py --dataset=RBC --architecture=Unet --symmetry=UM --output_length=4 --learning_rate=0.001 
python3 run_model.py --dataset=RBC --architecture=ResNet --symmetry=Rot --output_length=3 --learning_rate=0.001
python3 run_model.py --dataset=RBC --architecture=Unet --symmetry=Rot --output_length=4 --learning_rate=0.001
python3 run_model.py --dataset=RBC --architecture=ResNet --symmetry=Mag --output_length=3 --learning_rate=0.005
python3 run_model.py --dataset=RBC --architecture=Unet --symmetry=Mag --output_length=4 --learning_rate=0.005
python3 run_model.py --dataset=RBC --architecture=ResNet --symmetry=Scale --output_length=3 --learning_rate=0.0001
python3 run_model.py --dataset=RBC --architecture=Unet --symmetry=Scale --output_length=4 --learning_rate=0.0001

# Ocean Currents
# to reproduce the numbers, please run five times with different random seeds
python3 run_model.py --dataset=Ocean --architecture=ResNet --symmetry=None --output_length=4 --learning_rate=0.001 --input_length=24 --batch_size=32 --decay_rate=0.9 --seed=0
python3 run_model.py --dataset=Ocean --architecture=Unet --symmetry=None --output_length=5 --learning_rate=0.001 --input_length=21 --batch_size=16 --decay_rate=0.9 --seed=0
python3 run_model.py --dataset=Ocean --architecture=ResNet --symmetry=Rot --output_length=3 --learning_rate=0.001 --input_length=21 --batch_size=16 --decay_rate=0.9 --seed=0
python3 run_model.py --dataset=Ocean --architecture=Unet --symmetry=Rot --output_length=3 --learning_rate=0.001 --input_length=21 --batch_size=64 --decay_rate=0.9 --seed=0
python3 run_model.py --dataset=Ocean --architecture=ResNet --symmetry=UM --output_length=5 --learning_rate=0.001 --input_length=24 --batch_size=32 --decay_rate=0.9 --seed=0
python3 run_model.py --dataset=Ocean --architecture=Unet --symmetry=UM --output_length=5 --learning_rate=0.001 --input_length=24 --batch_size=32 --decay_rate=0.9 --seed=0
python3 run_model.py --dataset=Ocean --architecture=ResNet --symmetry=Mag --output_length=5 --learning_rate=0.001 --input_length=21 --batch_size=32 --decay_rate=0.9 --seed=0
python3 run_model.py --dataset=Ocean --architecture=Unet --symmetry=Mag --output_length=6 --learning_rate=0.001 --input_length=21 --batch_size=32 --decay_rate=0.9 --seed=0
python3 run_model.py --dataset=Ocean --architecture=ResNet --symmetry=Scale --output_length=3 --learning_rate=0.0001 --input_length=26 --batch_size=32 --decay_rate=0.9 --kernel_size=5 --seed=0 
python3 run_model.py --dataset=Ocean --architecture=Unet --symmetry=Scale --output_length=3 --learning_rate=0.0001 --input_length=26 --batch_size=16 --decay_rate=0.9 --kernel_size=5 --seed=0
