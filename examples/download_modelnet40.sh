export BASE_URL=http://cvgl.stanford.edu/data2/ModelNet40/

# set progress option accordingly
wget --help | grep -q '\--show-progress' && \
  _PROGRESS_OPT="-q --show-progress" || _PROGRESS_OPT=""

wget $_PROGRESS_OPT ${BASE_URL}/ModelNet40.tgz
tar -xzf ModelNet40.tgz
cd ModelNet40
wget ${BASE_URL}/train_modelnet40.txt
wget ${BASE_URL}/val_modelnet40.txt
wget ${BASE_URL}/test_modelnet40.txt