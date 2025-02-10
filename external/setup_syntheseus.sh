#!/bin/bash

# Install extra dependencies specific to Chemformer.
pip install pytorch-lightning==1.9.4 git+https://github.com/MolecularAI/pysmilesutils.git

export GITHUB_ORG_NAME=panpiort8
export GITHUB_REPO_NAME=syntheseus
export GITHUB_REPO_DIR=syntheseus
export GITHUB_COMMIT_ID=e5754963e4768ceada4fe7a60199bb1c5ceed2ea

source external/setup_shared.sh
