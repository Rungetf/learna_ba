.DEFAULT_GOAL := show-help
SHELL := /bin/bash
PATH := $(PWD)/thirdparty/miniconda/miniconda/bin:$(PATH)


################################################################################
# Utility
################################################################################

## Run the testsuite
test:
	@source activate learna && \
	pytest . -p no:warnings

## To clean project state
clean: clean-runtime clean-data clean-thirdparty clean-models clean-results clean-analysis

## Remove runtime files
clean-runtime:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '__pycache__' -exec rm -rf --force {} +

## Remove data files
clean-data:
	rm -rf data/eterna/*.rna
	rm -rf data/eterna/raw/*.txt
	rm -rf data/eterna/interim/*.txt
	# rm -rf data/rfam_taneda
	rm -rf data/rfam_learn*

## Remove model examples
clean-models:
	rm -rf models/example

## Clean results directory
clean-results:
	rm -rf results/

clean-plots:
	rm -rf results/plots

## Clean analysis directory
clean-analysis:
	rm -rf analysis/reproduce_iclr_2019/

## Remove thirdparty installs
clean-thirdparty:
	rm -rf thirdparty/miniconda/miniconda
	cd thirdparty/mcts && make clean
	cd thirdparty/rnainverse && make clean
	cd thirdparty/antarna && make clean
	cd thirdparty/eastman && make clean

################################################################################
# Setup General
################################################################################

## Download and prepare all datasets
data: data-eterna data-rfam-taneda data-rfam-learn

## Download and make the Eterna100 dataset
data-eterna:
	@source activate learna && \
	python -m src.data.download_and_build_eterna
	./src/data/secondaries_to_single_files.sh data/eterna data/eterna/interim/eterna.txt

## Download and build the Rfam-Taneda dataset
data-rfam-taneda:
	@./src/data/download_and_build_rfam_taneda.sh

## Download and build the Rfam-Learn dataset
data-rfam-learn:
	@./src/data/download_and_build_rfam_learn.sh
	mv data/rfam_learn/test data/rfam_learn_test
	mv data/rfam_learn/validation data/rfam_learn_validation
	mv data/rfam_learn/train data/rfam_learn_train
	rm -rf data/rfam_learn



################################################################################
# Setup LEARNA
################################################################################

## Install all dependencies
requirements:
	./thirdparty/miniconda/make_miniconda.sh
	conda env create -f environment.yml

## Install all thirdparty requirements
thirdparty-requirements:
	cd thirdparty/mcts && make requirements
	cd thirdparty/rnainverse && make requirements
	cd thirdparty/antarna && make requirements
	cd thirdparty/eastman && make requirements

################################################################################
# Test Experiment and Example
################################################################################

## Local experiment testing
experiment-test:
	@source activate learna && \
	python -m src.learna.design_rna \
	--mutation_threshold 5 \
  --batch_size 126 \
  --conv_sizes 17 5 \
  --conv_channels 7 18 \
  --embedding_size 3 \
  --entropy_regularization 6.762991409135427e-05 \
  --fc_units 57 \
  --learning_rate 0.0005991629320464973 \
  --lstm_units 28 \
  --num_fc_layers 1 \
  --num_lstm_layers 1 \
  --reward_exponent 9.33503385734547 \
  --state_radius 32 \
  --restart_timeout 1800 \
  --target_structure_path data/eterna/2.rna \
	--gc_improvement_step \
	--gc_reward \
	--desired_gc 0.1 \
	--gc_weight 1 \
	--gc_tolerance 0.01 \
	--timeout 600
	# --learning_rate 0.0005991629320464973 \


## Local Meta-LEARNA experiment with GC-control
meta-learna-test:
	@source activate learna && \
	python -m src.learna.design_rna \
	--mutation_threshold 5 \
	--batch_size 123 \
	--conv_sizes 11 3 \
	--conv_channels 10 3 \
	--embedding_size 2 \
	--entropy_regularization 0.00015087352506343337 \
	--fc_units 52 \
	--learning_rate 6.442010833400271e-05 \
	--lstm_units 3 \
	--num_fc_layers 1 \
	--num_lstm_layers 0 \
	--reward_exponent 8.932893783628236 \
	--state_radius 29 \
	--gc_improvement_step \
	--gc_reward \
	--desired_gc 0.1 \
	--gc_weight 1 \
	--gc_tolerance 0.01 \
	--target_structure_path data/eterna/2.rna \
	--restore_path models/ICLR_2019/224_0_1 \
	--stop_learning


## Local Meta-LEARNA experiment with GC-control
meta-learna-adapt-test:
	@source activate learna && \
	python -m src.learna.design_rna \
	--mutation_threshold 5 \
	--batch_size 123 \
	--conv_sizes 11 3 \
	--conv_channels 10 3 \
	--embedding_size 2 \
	--entropy_regularization 0.00015087352506343337 \
	--fc_units 52 \
	--learning_rate 6.442010833400271e-05 \
	--lstm_units 3 \
	--num_fc_layers 1 \
	--num_lstm_layers 0 \
	--reward_exponent 8.932893783628236 \
	--state_radius 29 \
	--gc_improvement_step \
	--gc_reward \
	--desired_gc 0.1 \
	--gc_weight 1 \
	--gc_tolerance 0.01 \
	--target_structure_path data/eterna/2.rna \
	--restore_path models/ICLR_2019/224_0_1 \
	--restart_timeout 1800



################################################################################
# Run experiments on Nemo cluster
################################################################################

## Reproduce LEARNA on <id> (1-100) of Eterna100
nemo-test-%:
	@source activate learna && \
	python utils/timed_execution.py \
		--timeout 60 \
		--data_dir data/ \
		--results_dir results/ \
		--experiment_group thesis_test \
		--method $* \
		--dataset rfam_taneda \
		--task_id 1

## Start experiment on the Rfam Taneda benchmark
nemo-rfam-taneda-%:
	msub utils/rna_single.moab \
		-l walltime=1200 \
		-t 1-1450 \
		-v METHOD=$* \
		-v DATASET=rfam_taneda \
		-v TIMEOUT=600 \
		-v EXPERIMENT_GROUP=thesis_LEARNA_gc_09


################################################################################
# Joint Architecture and Hyperparameter Search
################################################################################

## Run an example for joint Hyperparameter and Architecture Search using BOHB
bohb-example:
	@source activate learna && \
	python -m src.optimization.bohb \
	  --min_budget 30 \
		--max_budget 600 \
		--n_iter 1 \
		--n_cores 1 \
		--run_id example \
		--data_dir data \
		--nic_name lo \
		--shared_directory results/ \
		--mode learna

################################################################################
# Analysis and Visualization
################################################################################

## Analyse experiment group %
analyse-%:
	@source activate learna && \
	python -m src.analyse.analyse_experiment_group --experiment_group resutls/$* --analysis_dir analysis/$* --root_sequences_dir data --ci_alpha 0.05

## Analyse experiment group %
analyse-nemo-%:
	@source activate learna && \
	python -m src.analyse.analyse_experiment_group \
  	--experiment_group /work/ws/nemo/fr_ds371-learna-0/results/$* \
  	--analysis_dir /work/ws/nemo/fr_ds371-learna-0/analysis/$* \
  	--root_sequences_dir /work/ws/nemo/fr_ds371-learna-0/data
		cp -r /work/ws/nemo/fr_ds371-learna-0/analysis/$* analysis

## Plot reproduced results using pgfplots
plots-%:
	mkdir -p results
	rm -f results/plots.tex
	> results/plots.tex
	@source activate learna && \
	python -m src.analyse.plot --experiment_group analysis/$* --results_dir results/
	rm -rf results/plots/
	@source activate learna && \
	pdflatex -synctex=1 -interaction=nonstopmode -shell-escape results/plots.tex
	mkdir -p results/plots/
	mv plots.pdf results/plots/
	rm -f plots*
	okular results/plots/plots.pdf &



################################################################################
# Help
################################################################################

# From https://drivendata.github.io/cookiecutter-data-science/
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=22 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
