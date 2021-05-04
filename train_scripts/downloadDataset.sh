# download a dataset from the GCP bucket to the VM instance
echo "Downloading model and dataset from bucket..."
#create folder for model and train dataset
mkdir train_data_and_model
gsutil -m cp -r gs://<bucket>       # in <> add the dataset bucket 