if [ ! -d "./logs" ]; then
    mkdir ./logs
fi



root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

for model_name in DLinear
do
for seq_len in 96 192 336 512 720 1440
do
for pred_len in 24 48 96 192 336 512 720
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 24 \
    --enc_in 7 \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 32 --learning_rate 0.01
done
done
done

