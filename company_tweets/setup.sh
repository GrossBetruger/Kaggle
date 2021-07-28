#!/bin/bash

set -e 

python -m venv venv

. venv/bin/activate

pip install -r requirements.txt

python -m spacy download en_core_web_lg
