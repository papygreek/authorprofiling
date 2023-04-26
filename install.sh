#!/bin/bash
virtualenv -q -p /usr/bin/python3 $1
source $1/bin/activate
pip3 install -r requirements.txt