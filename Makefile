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

## Remove results directory
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

## Clean folder structure for rebuild
clean-analysis-env:
	@source activate learna && \
	if [ -d "$(PWD)/thirdparty/miniconda/miniconda/envs/analysis" ]; then \
	conda env remove -n analysis -y; \
	fi


################################################################################
# Setup General
################################################################################

## Download current rfam-database
download-rfam:
	./utils/data/download_rfam_database.sh

## Generate Rfam-interim dataset
rfam-interim:
	@source activate learna && \
	python -m src.data.generate_rfam_interim --family 6

rfam-local-dataset:
	@source activate learna && \
	python -m src.data.generate_rfam_dataset --unique --name rfam_local_long --size 100 --train_multiplier 1000 --validation_multiplier 1 --local_random --motifs

rfam-rna-design-local:
	@source activate learna && \
  python -m src.data.generate_rfam_taneda_local

split-local-data:
	@source activate learna && \
	python -m src.data.interim_to_single_files

validate-taneda-output:
	@source activate learna && \
	python -m src.analyse.validate_output


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

antarna-data:
	@source activate learna && \
	python -m src.data.generate_anta_data --data_path data --dataset rfam_local_test



################################################################################
# Setup LEARNA
################################################################################

## Install all dependencies
requirements:
	./thirdparty/miniconda/make_miniconda.sh
	conda env create -f environment.yml

## Install all dependencies for analysis
analysis-env:
	conda env create -f analysis.yml


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
	--agent random \
	--dataset 'test/test' \
	--data_dir 'data' \
	--target_structure_path 'data/riboswitch_design.rna' \
	--mutation_threshold 5 \
	--batch_size 196 \
	--conv_sizes 0 0 \
	--conv_channels 3 4 \
	--embedding_size 7 \
	--entropy_regularization 0.000528063083283936 \
	--fc_units 26 \
	--learning_rate 0.000384121783602744 \
	--lstm_units 8 \
	--num_fc_layers 1 \
	--num_lstm_layers 0 \
	--reward_exponent 11.454339469173966 \
	--state_radius 0 \
	--reward_function "structure_only" \
	--local_design \
	--state_representation "n-gram" \
	--predict_pairs \
	--data_type "random"

	# --restart_timeout 1800


## Local Meta-LEARNA experiment with GC-control
meta-learna-adapt-test:
	@source activate learna && \
	python -m src.learna.design_rna \
	--target_structure_path data/riboswitch_design.rna \
	--mutation_threshold 5 \
  --batch_size 95 \
  --conv_channels 23 2 \
  --conv_sizes 7 0 \
  --data_type "random" \
  --embedding_size 1 \
  --entropy_regularization 0.002846838967807086 \
  --fc_units 61 \
  --learning_rate 0.00035728046577710495 \
  --lstm_units 3 \
  --num_fc_layers 2 \
  --num_lstm_layers 0 \
  --predict_pairs \
  --reward_exponent 10.78740103642566 \
  --reward_function "structure_only" \
  --state_radius 22 \
  --state_representation "sequence_progress" \
  --local_design \
  --restore_path models/212_0_0/

meta-learna-test:
	@source activate learna && \
	python -m src.learna.design_rna \
	--target_structure_path data/riboswitch_design.rna \
	--mutation_threshold 5 \
	--batch_size 118 \
	--conv_channels 2 25 \
	--conv_sizes 0 5 \
	--data_type "random-sort" \
	--embedding_size 2 \
	--entropy_regularization 7.990216536104802e-05 \
	--fc_units 49 \
	--learning_rate 0.000854036796946988 \
	--lstm_units 1 \
	--num_fc_layers 1 \
	--num_lstm_layers 0 \
	--predict_pairs \
	--reward_exponent 10.892910865085273 \
	--reward_function "structure_only" \
	--state_radius 24 \
	--state_representation "sequence_progress" \
	--local_design \
	--restore_path models/471_0_8/
	--stop_learning


test-timed-execution-%:
	@source activate learna && \
	python utils/timed_execution.py \
		--timeout 600 \
		--data_dir data/ \
		--results_dir results/ \
		--experiment_group rna_local_design_new_interface \
		--method 7031046_330_0_3 \
		--dataset rfam_local_test_anta \
		--task_id $*

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
nemo-rfam-local-if-%:
	msub utils/rna_single.moab \
		-l walltime=6000 \
		-t 1-500 \
		-v METHOD=$* \
		-v DATASET=rfam_learn_test \
		-v TIMEOUT=3600 \
		-v EXPERIMENT_GROUP=partial_rna_design

## Start experiment on the Eterna100 benchmark
nemo-eterna-%:
	msub utils/rna_single.moab \
		-l walltime=100000 \
		-t 1-500 \
		-v METHOD=$* \
		-v DATASET=eterna \
		-v TIMEOUT=86400 \
		-v EXPERIMENT_GROUP=partial_rna_design

## Start experiment on the Rfam Taneda benchmark
nemo-rna-local-%:
	msub utils/rna_local.moab \
		-l walltime=6000 \
		-t 1-500 \
		-v METHOD=$* \
		-v DATASET=rfam_local_min_1000_test_gaps \
		-v TIMEOUT=3600 \
		-v EXPERIMENT_GROUP=partial_rna_design


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
		--mode meta_learna

################################################################################
# Analysis and Visualization
################################################################################
tables-%:
	@source activate learna && \
	python -m src.analyse.tables --experiment_group $* --results_dir tables/ --column_format "|c|c|c|c|c|c|c|c|c|" --compile_ready


validate-datasets:
	@source activate learna && \
	python -m src.data.validate_datasets --data_dir data

get-state-radius:
	@source activate learna && \
	python utils/get_state_radius.py --conv1 3 --conv2 0 --state_rel 0.4834547442899088

analyse-datasets:
	@source activate learna && \
	python -m src.analyse.analyse_datasets

verify:
	# rm -rf verification/rna_local_design_verify/*
	@source activate learna && \
	python -m src.analyse.verify_results


## Analyse experiment group %
analyse-performance-%:
	@source activate learna && \
	python -m src.analyse.analyse_experiment_group --experiment_group results/$* --analysis_dir analysis/$* --root_sequences_dir data --ci_alpha 0.05

## Analyse experiment group %
analyse-performance-nemo-%:
	@source activate learna && \
	python -m src.analyse.analyse_experiment_group \
  	--experiment_group /work/ws/nemo/fr_ds371-learna-0/results/$* \
  	--analysis_dir /work/ws/nemo/fr_ds371-learna-0/analysis/$* \
  	--root_sequences_dir /work/ws/nemo/fr_ds371-learna-0/data
		cp -r /work/ws/nemo/fr_ds371-learna-0/analysis/$* analysis

## Analyse the output
analyse-output-%:
	@source activate learna && \
	python -m src.analyse.analyse_output --experiment_group /home/fred/research/thesis/results_raw/thesis_new_output_gc_0$*

## Analyse the output
analyse-output-test:
	@source activate learna && \
	python -m src.analyse.analyse_output --experiment_group /home/fred/research/thesis/test_results/

## Analyse Bohb runs
analyse-bohb-%:
	@source activate analysis && \
	python -m src.analyse.analyse_bohb_results --run $* --out_dir results/fanova/$* --mode autoMeta-LEARNA --n 5 --parameter reward_function


generate-bohb-plotting-data-%:
	@source activate learna && \
	python -m src.analyse.generate_bohb_pgfplot_data --path results/bohb --run $* --out_dir analysis/bohb



## Plot reproduced results using pgfplots
plots-%:
	mkdir -p results
	mkdir -p results/plots/
	rm -f results/plots.*
	rm -f results/plots/$*.*
	> results/plots.tex
	@source activate learna && \
	python -m src.analyse.plot --experiment_group analysis/$* --results_dir results/
	@source activate learna && \
	pdflatex -synctex=1 -interaction=nonstopmode -shell-escape results/plots.tex
	mv plots.pdf results/plots/$*.pdf
	rm -f plots*
	okular results/plots/$*.pdf &

## Plot gc-control data
final-plots:
	rm -f pgfplots/*
	pdflatex -synctex=1 -interaction=nonstopmode -shell-escape pgfplots.tex

other-plots:
	mkdir -p other_plots
	pdflatex -synctex=1 -interaction=nonstopmode -shell-escape other_pgf_plots.tex


## Plot gc-control data
performance-plots:
	rm -f pgfplots_performance/*
	pdflatex -synctex=1 -interaction=nonstopmode -shell-escape pgfplots_performance.tex

write_config_space:
	@source activate learna && \
	python -m src.analyse.write_config_space

analyse-evaluations:
	@source activate learna && \
	python -m src.analyse.analyse_number_of_evaluations --experiment_group results/partial_rna_design/ --methods 7052569_471_0_8 7052570_212_0_0 7052571_188_0_3 --data_dir data --out_dir analysis


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
