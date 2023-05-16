FROM fastdotai/fastai:2020-10-02

RUN useradd fastai-user 

RUN apt-get update 

# RUN apt-get -y install nano\
#    graphviz \
#    libwebp-dev

WORKDIR /fged

COPY . /fged

RUN pip install -r requirements.txt

CMD ["python" , "functions.py"]