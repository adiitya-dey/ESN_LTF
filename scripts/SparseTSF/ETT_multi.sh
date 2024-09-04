if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=ESN
seq_len=336
root_path_name=./dataset/

for data_name in ETTh1 ETTh2 ETTm1 ETTm2
do 
model_id_name=$data_name
data_path_name=$data_name'.csv'
for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features S \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 24 \
    --enc_in 1 \
    --train_epochs 30 \
    --patience 5 \
    --individual 1 \
    --loss mse \
    --itr 1 --batch_size 32 --learning_rate 0.02
done
done
