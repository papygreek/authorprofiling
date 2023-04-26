import subprocess

# Path to a Python interpreter that runs any Python script
# under the virtualenv /path/to/virtualenv/
python_bin = "./.venv.nosync/bin/python3"

# Path to the script that must run under the virtualenv
script_file = "./cli.py"

subprocess.Popen([python_bin, script_file])