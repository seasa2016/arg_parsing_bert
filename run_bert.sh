data=('changemyview' 'total' 'argumentessay')
#data=('argumentessay')

for((i=0;$i<${#data[@]};i=i+1))
do
	echo ${data[i]}
	python run_crf.py --task_name arg --do_train --do_lower_case   \
		--data_dir ./../preprocess/parsing/${data[i]}   --bert_model bert-base-uncased  --max_seq_length 400   \
		--train_batch_size 8   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir ./saved_models/${data[i]}_topic

	#for((j=0;$j<${#data[@]};j=j+1))
	#do
	#	echo ${data[i]}_${data[i]}
	#	python run_crf.py --task_name arg --do_eval --do_lower_case   \
	#		--data_dir ./../preprocess/${data[i]}   --bert_model ./saved_models/${data[i]}_crf  --max_seq_length 400 --output_dir ./saved_models/${data[i]}_crf

		#echo ${data[i]}_${data[j]} >> result.txt
		#python eval.py ./saved_models/${data[i]}_crf/pred.txt >> result.txt
	#done
done




