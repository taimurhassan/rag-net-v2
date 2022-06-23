FROM nvcr.io/nvidia/tensorflow:21.08-tf2-py3

# install python requirements
RUN pip install pip==21.0.1
pip uninstall -y tensorboard tb-nightly &&
pip install tb-nightly  # must have at least tb-nightly==2.5.0a20210316
tensorboard --logdir /path/to/logs --load_fast true
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# add the source code
WORKDIR /
ADD . .
