#!/bin/bash
python3 -m venv .venv.nosync --prompt authorprofiling
source ./.venv.nosync/bin/activate
pip3 install -r requirements.txt
pip3 uninstall -y adjusttext
pip3 install https://github.com/Phlya/adjustText/archive/master.zip
chmod +x run.sh