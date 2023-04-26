#!/bin/bash
python3 -m venv .venv.nosync --prompt authorprofiling
source ./.venv.nosync/bin/activate
pip3 install -r requirements.txt