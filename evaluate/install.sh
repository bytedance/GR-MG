#!/bin/bash
# You should first install the packages needed by policy ang goal image generation model
# install calvin
cd /PATH/TO/GR_MG
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=$(pwd)/calvin
cd calvin
cd calvin_env; git checkout main
cd ..

cd $CALVIN_ROOT
pip3 install setuptools==57.5.0
sh install.sh
cd /PATH/TO/GR_MG
export EVALUTION_ROOT=$(pwd)

# Install dependency for calvin
sudo apt-get -y install libegl1-mesa libegl1
sudo apt-get -y install libgl1
sudo apt-get -y install libosmesa6-dev
sudo apt install ffmpeg
sudo apt-get -y install patchelf