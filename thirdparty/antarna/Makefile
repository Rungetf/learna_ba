.PHONY: clean data lint requirements show-help
.DEFAULT_GOAL := show-help
SHELL := /bin/bash
PATH := $(PWD)/../miniconda/miniconda/bin:$(PATH)

################################################################################
# GLOBALS
################################################################################


################################################################################
# Utility
################################################################################

## Clean folder structure for rebuild
clean:
	if [ -d "$(PWD)/../miniconda/miniconda/envs/antarna" ]; then \
	conda env remove -n antarna -y; \
	fi
	rm -rf src/


################################################################################
# Setup
################################################################################

## Make all requirements
requirements: requirements-conda requirements-python requirements-vienna

## Create Conda Environment for antaRNA
requirements-conda:
	conda create --no-default-packages -y --name antarna python=2.7


## Install all requirements and download antaRNA source
requirements-python:
	./install_antarna.sh

## Install ViennaRNA
requirements-vienna:
	conda install --name antarna -y -c bioconda viennarna=2.1.9


################################################################################
# Local Experiments
################################################################################

## Local experiment testing
experiment-test:
	@source activate antarna && \
	python src/antaRNA_v114.py \
	  -Cstr "...(((.....)))...(((((......))..))).." \
		-tGC 0.5 \
		-n 1 \
		-noLBP \
		-r 2147483647 \
		-t 2147483647 \
		-Cgcw 0.0 \
		-ov

		# -aTC 2147483647 \




################################################################################
# Nemo Cluster Experiments
################################################################################




################################################################################
# Make Selfdocumentation
################################################################################
# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
################################################################################

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
