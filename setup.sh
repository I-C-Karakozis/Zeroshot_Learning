# setup slurm scripts
mkdir slurm_scripts
python generate_slurm_scripts.py

# setup directories
mkdir figures
mkdir logging
mkdir models
mkdir predictions

# get data
mkdir data
cd data
wget https://www.dropbox.com/s/33v1s9ri85o21x7/JPEGImages_128x128.zip
unzip JPEGImages_128x128.zip
mv JPEGImages_128x128 AwA_128x128
cd ..
