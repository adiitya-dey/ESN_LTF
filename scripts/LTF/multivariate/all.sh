if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=LTF

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

rank=14

for pred_len in 96 192 336 720
do   
for seq_len in 336 512
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
      --enc_in 7 \
      --train_epochs 100 \
      --rank $rank \
      --patience 20 \
      --itr 1 --batch_size 32 --learning_rate 0.01
done
done

root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

rank=24

for pred_len in 96 192 336 720
do   
for seq_len in 336 512
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
      --enc_in 7 \
      --train_epochs 100 \
      --rank $rank \
      --patience 20 \
      --itr 1 --batch_size 32 --learning_rate 0.01
done
done

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1


rank=24


for pred_len in 96 192 336 720
do 
for seq_len in 336 512
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
      --enc_in 7 \
      --train_epochs 100 \
      --rank $rank \
      --patience 20 \
      --itr 1 --batch_size 32 --learning_rate 0.01
done
done

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2


rank=24

for pred_len in 96 192 336 720
do   
for seq_len in 336 512
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
      --enc_in 7 \
      --train_epochs 100 \
      --rank $rank \
      --patience 20 \
      --itr 1 --batch_size 32 --learning_rate 0.01
done
done

model_name=LTF

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

rank=24

for pred_len in 96 192 336 720
do   
for seq_len in 336 512
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
    --enc_in 21 \
    --rank $rank \
    --train_epochs 100 \
    --patience 20 \
    --itr 1 --batch_size 32 --learning_rate 0.01
done
done