#!/bin/bash
python3 -m venv .venv.nosync --prompt authorprofiling
source ./.venv.nosync/bin/activate
pip3 install -r requirements.txt
chmod +x run.sh
curl -o ./authorprofiling/fasttext-ancientgreek.bin "https://www.icloud.com/iclouddrive/02aFL71pzyL99vhVr8lI-5THw#fasttext-ancientgreek"
curl -o ./authorprofiling/word2vec-ancientgreek.bin "https://www.icloud.com/iclouddrive/088L-ppQYAKf1Xb9naw2ecUFA#word2vec-ancientgreek"