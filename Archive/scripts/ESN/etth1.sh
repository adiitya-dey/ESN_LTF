# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=ESN
data=ETTh1
data_path=ETTh1.csv
seq_len=336
pred_len=96
window_len=12
reservoir_size=5
washout=10
model_id=$data'_'$seq_len'_'$pred_len



python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $data_path \
  --model_id $model_id \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --window_len $window_len \
  --reservoir_size $reservoir_size \
  --washout $washout \
  --enc_in 7 \
  --individual 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'$model_id.log
