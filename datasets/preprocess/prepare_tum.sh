# Reference: https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md, https://github.com/Junyi42/monst3r/blob/main/data/download_tum_dynamics.sh
# The scripts provided here are for reference only. Please ensure you have obtained the necessary licenses from the original dataset providers before proceeding.

# download tum-dynamic dataset
mkdir -p data/tum
cd data/tum

wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_static.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_xyz.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_halfsphere.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_rpy.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_static.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_halfsphere.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_rpy.tgz

# unzip all
find . -name "*.tgz" -exec tar -zxvf {} \;
# remove all zip files
find . -name "*.tgz" -exec rm {} \;

# prepare the dataset for camera evaluation
python datasets/preprocess/prepare_tum.py