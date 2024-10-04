# @File  : run.py
# @Author: fkx
# @Time: 2024/3/6 14:01 
## -*- coding: utf-8 -*-
#list1=('Fish')
#list2=('sleepEDF' 'Worms' 'Haptics' 'SemgHandSubjectCh2' 'ECG5000' 'Plane' 'Lightning7' 'Fish')
#### 循环输出列表中的元素
#for directory in "${list1[@]}"
#do
#python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode self_supervised --model_select mlp --selected_dataset "$directory" --instance_loss False --fuzzy False
##python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode train_linear --model_select mlp --selected_dataset "$directory" --instance_loss False --fuzzy False
##
##python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode self_supervised --model_select DilatedConv --selected_dataset "$directory" --instance_loss False --fuzzy True
##python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode train_linear --model_select DilatedConv --selected_dataset "$directory" --instance_loss False --fuzzy True
#
#done
#python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode self_supervised --model_select mlp --selected_dataset "$directory" --instance_loss False --fuzzy True
#python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode train_linear --model_select mlp --selected_dataset "$directory" --instance_loss False --fuzzy True
#
#python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode self_supervised --model_select DilatedConv --selected_dataset "$directory" --instance_loss False --fuzzy True
#python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode train_linear --model_select DilatedConv --selected_dataset "$directory" --instance_loss False --fuzzy True
#done

# ==================================mydata_aldata===============================

##  python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode self_supervised --model_select mlp --selected_dataset Aldata --pot_period "$filename" --instance_loss False --fuzzy False --features_len $i
##  python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode train_linear --model_select mlp --selected_dataset Aldata --pot_period "$filename" --instance_loss False --fuzzy False --features_len $i
#  python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode self_supervised --model_select mlp --selected_dataset Aldata --pot_period "$filename" --instance_loss False --fuzzy True --features_len $i
#  python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode train_linear --model_select mlp --selected_dataset Aldata --pot_period "$filename" --instance_loss False --fuzzy True --features_len $i
##  python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode self_supervised --model_select DilatedConv --selected_dataset Aldata --pot_period "$filename" --instance_loss False --fuzzy True --features_len $i
##  python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode train_linear --model_select DilatedConv --selected_dataset Aldata --pot_period "$filename" --instance_loss False --fuzzy True --features_len $i
#}
#
#process_file "$file_56"
#python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode self_supervised --model_select mlp --selected_dataset Aldata --pot_period 5228_2160_5 --instance_loss False --fuzzy False
#python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode train_linear --model_select mlp --selected_dataset Aldata --pot_period 5228_2160_5 --instance_loss False --fuzzy False
#python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode self_supervised --model_select mlp --selected_dataset Aldata --pot_period 5228_2160_5 --instance_loss False --fuzzy True
#python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode train_linear --model_select mlp --selected_dataset Aldata --pot_period 5228_2160_5 --instance_loss False --fuzzy True

python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode self_supervised --model_select DilatedConv --selected_dataset Aldata --pot_period 5228_2160_5 --instance_loss False --fuzzy True
python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode train_linear --model_select DilatedConv --selected_dataset Aldata --pot_period 5228_2160_5 --instance_loss False --fuzzy True






