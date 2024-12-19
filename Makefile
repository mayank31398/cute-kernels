install:
	pip install .

install-dev:
	pip install -e .

test:
	pytest tests

update-precommit:
	pre-commit autoupdate

style:
	pre-commit run --all-files

cutotune-cache:
	LOAD_CUTOTUNE_CACHE=0 TORCH_CUDA_ARCH_LIST=9.0 python tools/build_cutotune_cache.py
