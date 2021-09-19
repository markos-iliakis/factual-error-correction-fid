model=$1
lr=$2
masker=$3
mutation_source=false
mutation_target=false
labels=$4
data=$5
bs=${6:-4}
seed=${SEED:-42}

output_dir=output/masker/model=${model},lr=${lr},masker=${masker},data=${data},mutation_source=${mutation_source},mutation_target=${mutation_target},labels=${labels}/seed=$seed/

if [ $6 == 'train' ]; then
  echo 'training'
  python -m error_correction.corrector.run \
    --model_name_or_path $model \
    --output_dir $output_dir \
    --train_file /content/drive/MyDrive/Colab\ Notebooks/factual_error_correction/resources/intermediate\ masks/${masker}_train_${data}.jsonl \
    --val_file /content/drive/MyDrive/Colab\ Notebooks/factual_error_correction/resources/intermediate\ masks/${masker}_dev_${data}.jsonl \
    --do_train \
    --train_batch_size 1 \
    --learning_rate $lr \
    --seed $seed \
    --num_train_epochs 4 \
    --reader mask \
    --mutation_source ${mutation_source} \
    --mutation_target ${mutation_target} \
    --labels ${labels} \
    --eval_batch_size 1

elif [ $6 == 'test' ]; then
  echo 'testing'
  python -m error_correction.corrector.run \
    --model_name_or_path t5-small \
    --output_dir $output_dir \
    --do_predict \
    --train_file /content/drive/MyDrive/Colab\ Notebooks/factual_error_correction/resources/intermediate\ masks/heuristic_ir_train_${data}.jsonl \
    --val_file /content/drive/MyDrive/Colab\ Notebooks/factual_error_correction/resources/intermediate\ masks/heuristic_ir_dev_${data}.jsonl \
    --test_file /content/drive/MyDrive/Colab\ Notebooks/factual_error_correction/resources/intermediate\ masks/heuristic_ir_test_${data}.jsonl \
    --reader mask \
    --mutation_source true \
    --mutation_target false \
    --labels all \
    --eval_batch_size 1
fi



