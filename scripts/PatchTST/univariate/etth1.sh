if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=LTF

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1


seq_len=512
for pred_len in 24 48 96 192
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
    --period_len 12 \
    --enc_in 1 \
    --train_epochs 30 \
    --patience 5 \
    --individual 1 \
    --loss mse \
    --itr 1 --batch_size 32 --learning_rate 0.01
done

