#!/bin/bash
cd data/iNat2018

echo "Downloading iNaturalist 2018 train/val..."
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train2018.json.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/val2018.json.tar.gz

echo "Extracting..."
tar -xzvf train_val2018.tar.gz
tar -xzvf train2018.json.tar.gz
tar -xzvf val2018.json.tar.gz
