if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi


model_name=ESN
data=ETTm1
data_path=ETTm1.csv
seq_len=336
window_len=12
reservoir_size=5
washout=10
model_id=$data'_'$seq_len'_'$pred_len



# ETTh1, univariate results, pred_len= 24 48 96 192 336 720

random_seed=2021
for pred_len in 24 48 96 192 336 720
do
  model_id=$data'_'$seq_len'_'$pred_len
  python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path $data_path \
    --model_id $model_id \
    --model $model_name \
    --data $data \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --window_len $window_len \
    --reservoir_size $reservoir_size \
    --washout $reservoir_size \
    --loss 'mse' \
    --enc_in 1 \
    --des 'Exp' \
    --itr 1 --batch_size 32 --feature S --learning_rate 0.01 >logs/LongForecasting/$model_name'_'$model_id.log
done 