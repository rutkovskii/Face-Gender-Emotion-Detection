FROM symeneses/fastai

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y cmake
RUN pip install dlib==19.24.1

VOLUME /fged

WORKDIR /fged

RUN apt-get update && apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-alsa \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav

RUN apt-get install -y sudo

# COPY . /fged
COPY build_opencv.sh ./build_opencv.sh
COPY requirements.txt ./requirements.txt

RUN chmod +x build_opencv.sh
RUN /bin/bash build_opencv.sh

RUN pip install -r requirements.txt

CMD ["/bin/bash"]


