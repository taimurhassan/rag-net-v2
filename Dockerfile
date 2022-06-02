FROM registry.kao.instadeep.io/library/nvidia/tensorflow:21.08-tf2-py3

# install python requirements
RUN pip install pip==21.0.1
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# add the source code
WORKDIR /
ADD . .
