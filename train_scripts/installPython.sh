#!/bin/bash
#first update system
echo "system update..."
sudo apt-get update
echo "system updated"
#install needed software properties
echo "installing prerequiesties..."
sudo apt install software-properties-common -y
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl libbz2-dev
#download tar for python
echo "Downloading Python 3.8.2..."
curl -O https://www.python.org/ftp/python/3.8.2/Python-3.8.2.tar.xz
#unpack tar
echo "Unpacking Python 3.8.2..."
tar -xf Python-3.8.2.tar.xz
cd Python-3.8.2
./configure --enable-optimizations

#install python 3.8.2
echo "Installing Python 3.8.2..."
make altinstall

#install python packages using pip
echo "Installing GCP needed packages..."
sudo apt-get install -y python3-pip
sudo apt-get update
sudo apt-get install google-cloud-sdk -y
sudo apt-get install python3-venv -y
#echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
#apt-get install apt-transport-https ca-certificates gnupg -y
#curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
#apt-get update && sudo apt-get install google-cloud-sdk -y
#gcloud init

#go back and delete Python folder
echo "Removing Python 3.8.2 folder..."
cd ..
ls
rm -rf Python-3.8.2
rm -rf Python-3.8.2.tar.xz

