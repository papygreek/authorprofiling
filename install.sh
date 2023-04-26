#!/bin/bash
python3 -m venv .venv.nosync --prompt authorprofiling
source ./.venv.nosync/bin/activate
pip3 install -r requirements.txt
chmod +x run.sh
curl -o ./authorprofiling/fasttext-ancientgreek.bin "https://drive.google.com/uc?export=download&id=11f7A8-GsYBkzXlusYtKiRuMDdUem0Klk"
curl -o ./authorprofiling/word2vec-ancientgreek.bin "https://drive.google.com/uc?export=download&id=1aSlhh_BcSCbQv3Icj3LQT3QIoysfdJo6"