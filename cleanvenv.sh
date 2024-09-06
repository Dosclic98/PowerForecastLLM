#!/bin/sh
# Uninstall all python packages within the venv 
# to prevent this bug from happening
# https://github.com/horovod/horovod/issues/4042
pip freeze | xargs pip uninstall -y