# source this: sets data/cache locations on the shared space
export ACMEC_DATA_ROOT=/net/spaces/jevans/klab/caroyf/data/ACMEC
export HF_HOME=$ACMEC_DATA_ROOT/hf_cache
export PIP_CACHE_DIR=/net/spaces/jevans/klab/caroyf/conda/pip_cache
export PYTHONPATH=/home/caroyf/myproject/ACMEC/src${PYTHONPATH:+:$PYTHONPATH}
