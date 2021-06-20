## Code for bio parsing
python preprocess_parsing.py "input file" "output folder"
### training
python train.py --model_name_or_path bert-base-uncased --data_dir {PATH_TO_DATA} --do_train --do_eval --learning_rate 5e-5 --num_train_epochs 3.0 --output_dir ./saved_models/gaku_essay_55/
### testing
python train.py --bert_model {PATH_TO_CHECKPOINT_FOLDER} --data {PATH_TO_DATA} --output_dir {MODEL_FOLDER} --pred_name {PREDICT_PATH} --do_test

