FROM nvcr.io/nvidia/tensorflow:22.05-tf1-py3

# install python requirements
RUN pip install pip==21.0.1
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# add the source code
WORKDIR /
ADD . . 
