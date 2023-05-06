.ONESHELL:

.PHONY: install

PROJECT?=poirot
VERSION?=3.11.3
VENV=${PROJECT}-${VERSION}
VENV_DIR=$(shell pyenv root)/versions/${VENV}
PYTHON=${VENV_DIR}/bin/python
JUPYTER_ENV_NAME=${VENV}
GLOBAL=
INLCUDE_CPATH=/opt/homebrew/include/
INCLUDE_HDF5_DIR=/opt/homebrew/opt/hdf5/1.12.2_2

CVERSION?=3.10
CVENV?=conda-${PROJECT}-${CVERSION}
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate


install:
	@echo "Installing $(VENV)"
	env PYTHON_CONFIGURE_OPTS=--enable-shared pyenv virtualenv ${VERSION} ${VENV}
	pyenv local ${VENV}
	$(PYTHON) -m pip  install -U pip setuptools wheel flit
	export CPATH=${CPATH}:${INLCUDE_CPATH}; export HDF5_DIR=${HDF5_DIR}:${INCLUDE_HDF5_DIR}; $(PYTHON) -m pip install h5py --no-cache-dir
	export HDF5_DIR=/opt/homebrew/opt/hdf5; export BLOSC_DIR=/opt/homebrew/opt/c-blos ; $(PYTHON) -m pip install tables
	$(PYTHON) -m pip install -U git+https://github.com/fooof-tools/fooof.git
	$(PYTHON) -m pip install -U git+https://github.com/danieltomasz/python-ggseg.git
   	$(PYTHON) -m pip install -U git+https://github.com/pyxnat/pyxnat.git@bbrc
	$(PYTHON) -m flit install --symlink
	$(PYTHON) -m pip install  -r requirements.txt
	$(PYTHON) -m ipykernel install --user --name ${VENV}

update:
	$(PYTHON) -m pip install --upgrade -r requirements.txt --upgrade-strategy=eager
	$(PYTHON) -m pip install -U git+https://github.com/fooof-tools/fooof.git
purge:
	$(PYTHON) -m pip cache purge


uninstall:
	@echo "Removing $(VENV)"
	-jupyter kernelspec uninstall ${VENV}
	-pyenv uninstall ${VENV}
	-rm .python-version
	-rm -r /Users/daniel/Library/Jupyter/kernels/${VENV}


outdated:
	pyenv local ${VENV}
	$(PYTHON) -m pip list --outdated

kernels:
	jupyter kernelspec list

conda-activate:
	$(CONDA_ACTIVATE) ${CVENV}

conda-install:
	conda env create --name ${CVENV} --file local.yml 
	$(CONDA_ACTIVATE) ${CVENV}; flit install --env --symlink --deps none
	$(CONDA_ACTIVATE) ${CVENV};pip uninstall grpcio; conda install grpcio 
	$(CONDA_ACTIVATE) ${CVENV}; ipython kernel install --user --name=${CVENV}

conda-update:
	$(CONDA_ACTIVATE) ${CVENV}; conda env update --name ${CVENV} --file local.yml --prune

conda-remove:
	conda env remove --name ${CVENV} 

conda-reinstall: conda-remove conda-install

conda-list:
	conda list -n ${CVENV}