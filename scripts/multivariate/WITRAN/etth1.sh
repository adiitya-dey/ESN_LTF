if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=WITRAN

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

for seq_len in 512
do
for pred_len in 96 192 336 720
do    
    python -u run_longExp.py \
      --is_training 1 \
      --individual 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --train_type nonlinear \
      --enc_in 7 \
      --c_out 7 \
      --train_epochs 1 \
      --WITRAN_deal standard \
      --WITRAN_grid_cols 32 \
      --e_layers 4 \
      --d_model 16 \
      --patience 20 \
      --des 'Exp' \
      --itr 1 --batch_size 32 --learning_rate 0.01
done
done

