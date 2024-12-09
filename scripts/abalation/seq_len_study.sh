if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=HaarDCT

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

rank=14

for seq_len in 48 96 192 336 512 720
do
for pred_len in 96 192 336 720
do    
    python -u abalation_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --train_epochs 100 \
      --rank $rank \
      --patience 20 \
      --des 'Exp' \
      --itr 1 --batch_size 32 --learning_rate 0.01
done
done

