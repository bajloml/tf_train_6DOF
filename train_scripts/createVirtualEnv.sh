#create virtual enviroment
echo "Creating virtual enviroment..."
mkdir env
cd env
/usr/local/bin/python3.8 -m venv venv
cd venv/bin/
. activate

# upgrade pip in the venv
pip install --upgrade pip

#install packages and libraries needed for the model
echo "Installing gsutil..."
sudo apt-get install gcc python-dev python-setuptools libffi-dev -y
sudo apt-get install python-pip -y
pip install gsutil -y
echo "Installing packages needed for the model..."
# pip3 install tensorflow
pip3 install tensorflow==2.4.0
pip3 install numpy
python -m pip install -U matplotlib
pip3 install pandas
pip3 install configparser
pip install scikit-build
pip3 install opencv-python
pip3 install pydot
pip3 install graphviz

sudo apt-get install graphviz

#uninstalling packages (just for test)
#echo "Uninstalling packages from enviroment..." 
#pip3 uninstall configparser -y
#pip3 uninstall tensorflow -y
#pip3 uninstall pandas -y
#pip3 uninstall numpy -y

#deactivate enviroment
. deactivate
