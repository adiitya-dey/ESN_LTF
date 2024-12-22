if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=FreTS

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

for noise_std in 0.0 0.1 0.3 0.5 0.7 
do
for seq_len in 512
do
for pred_len in 96 192 336 720
do    
    python -u run_noisetesting.py \
      --is_training 1 \
      --individual 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M --train_type Linear \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --train_epochs 100 \
      --patience 20 \
      --des 'Exp' \
      --itr 1 --batch_size 32 --learning_rate 0.01
done
done

done
