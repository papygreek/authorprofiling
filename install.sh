#!/bin/bash
python3 -m venv .venv.nosync --prompt authorprofiling
source ./authorprofiling/bin/activate
pip3 install -r requirements.txt