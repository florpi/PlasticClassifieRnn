 #!/bin/bash

# Hide GPU from tensorflow
# export CUDA_VISIBLE_DEVICES=-1
#!/bin/bash
set -e
# Hide GPU from tensorflow
# export CUDA_VISIBLE_DEVICES=-1

# Train and evaluate
for i in {1..5}
do
	if [ ! -d train/fold_"$i"_of_4 ]
	then
		mkdir -p train/fold_"$i"_of_4
	fi
	python3 classifier/model_main.py \
	--model "RNN" \
	--model_dir train/fold_"$i"_of_4/ \
	--dataset_dir records/ \
	--batch_size 150 \
	--num_epochs 400 \
	--learning_rate 1e-3 \
	--validation_fold $i
done

