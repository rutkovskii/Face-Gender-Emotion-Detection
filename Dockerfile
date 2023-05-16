FROM fastdotai/fastai:2020-10-02

RUN #!/bin/bash apt-get update 

# RUN apt-get -y install nano\
#    graphviz \
#    libwebp-dev

WORKDIR /fged

COPY . /fged

RUN #!/bin/bash pip install -r requirements.txt

CMD ["python" , "functions.py"]