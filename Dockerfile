# FROM fastai/

FROM symeneses/fastai

RUN #!/bin/bash apt-get update 

# RUN apt-get -y install nano\
#    graphviz \
#    libwebp-dev

WORKDIR /fged

COPY . /fged

RUN #!/bin/bash pip install -r requirements.txt

CMD ["python" , "functions.py"]