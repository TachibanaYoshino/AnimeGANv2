FROM python:3.7-bookworm

RUN apt-get update
# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get install ffmpeg libsm6 libxext6  -y

ARG UNAME=animegan

WORKDIR /home/$UNAME
RUN mkdir -p /home/$UNAME/dataset
RUN mkdir -p /home/$UNAME/results
RUN mkdir -p /home/$UNAME/video

RUN pip install tensorflow-gpu==1.15.0 \
    opencv-python \
    tqdm \
    numpy \
    argparse \
    onnxruntime

# Downgrade protobuf - see https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
RUN pip install protobuf==3.20.*

## (WIP) If you care about security and want use non-root user inside container, uncomment this before build. But it may make your life harder in some ways.
# ARG UID=1000
# ARG GID=1000
# RUN groupadd -g $GID -o $UNAME
# RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
# RUN chown -cR $UNAME /home/$UNAME
# RUN chown -cR $UNAME /home/$UNAME/dataset 
# RUN chown -cR $UNAME /home/$UNAME/results 
# RUN chown -cR $UNAME /home/$UNAME/video 
# USER $UNAME
## Build command start like that: docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) 

COPY . .

ENTRYPOINT ["/bin/bash"]