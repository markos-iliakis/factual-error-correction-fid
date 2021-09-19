model=$1
lr=$2
mutation_source=$3
mutation_target=$4
labels=$5
data=$6
bs=${7:-4}
seed=${SEED:-42}

output_dir=output/supervised/model=${model},reader=${reader},data=${data},lr=${lr},mutation_source=${mutation_source},mutation_target=${mutation_target},labels=${labels}/seed=$seed/

if [ $7 == 'train' ]; then
  echo 'training'
  python -m error_correction.corrector.run \
    --model_name_or_path $model \
    --output_dir $output_dir \
    --reader supervised \
    --train_file /content/drive/MyDrive/Colab\ Notebooks/factual_error_correction/resources/retrieval/genre_50/${data}_train.jsonl \
    --val_file /content/drive/MyDrive/Colab\ Notebooks/factual_error_correction/resources/retrieval/genre_50/${data}_dev.jsonl \
    --do_train \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --learning_rate $lr \
    --seed $seed \
    --num_train_epochs 4 \
    --mutation_source ${mutation_source} \
    --mutation_target ${mutation_target} \
    --labels ${labels}

elif [ $7 == 'test' ]; then
  echo 'testing'
  python -m error_correction.corrector.run \
    --model_name_or_path t5-small \
    --output_dir $output_dir \
    --do_predict \
    --train_file /content/drive/MyDrive/Colab\ Notebooks/factual_error_correction/resources/retrieval/dpr_50/dpr_50_2_train.jsonl \
    --val_file /content/drive/MyDrive/Colab\ Notebooks/factual_error_correction/resources/retrieval/dpr_50/dpr_50_2_dev.jsonl \
    --test_file /content/drive/MyDrive/Colab\ Notebooks/factual_error_correction/resources/retrieval/dpr_50/dpr_50_2_test.jsonl \
    --reader supervised \
    --mutation_source true \
    --mutation_target false \
    --labels all \
    --eval_batch_size 1
fi