.PHONY: env
env:
	conda env update -f environment.yml

.PHONY: train
train:
	python train.py --config "configs/train.cfg"

.PHONY: eval
eval:
	python eval.py --model_dir "$(MODEL_DIR)"
