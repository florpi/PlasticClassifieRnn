 #!/bin/bash

# Hide GPU from tensorflow
# export CUDA_VISIBLE_DEVICES=-1
#!/bin/bash
set -e
# Hide GPU from tensorflow
# export CUDA_VISIBLE_DEVICES=-1

model=$1

echo "Training RNN using" "$model" "features. Good luck !" 

# Train and evaluate
for i in {1..5}
do
	if [ ! -d train_"$model"/fold_"$i"_of_4 ]
	then
		mkdir -p train_"$model"/fold_"$i"_of_4
	fi
	python3 classifier/model_main.py \
	--model "$model" \
	--model_dir train_"$model"/fold_"$i"_of_4/ \
	--dataset_dir tfrecords/ \
	--batch_size 150 \
	--num_epochs 400 \
	--learning_rate 1e-3 \
	--validation_fold $i 
done

