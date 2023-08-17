FROM python:3.7-bookworm

RUN useradd -ms /bin/bash animegan

RUN apt-get update
# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /anime_gan_dir
RUN mkdir -p /anime_gan_dir/dataset
RUN mkdir -p /anime_gan_dir/results
RUN mkdir -p /anime_gan_dir/video

## If you care about security, you can use non-root user inside container, but it will make your life harder in some ways.
# ARG UNAME=animegan
# ARG UID=1000
# ARG GID=1000
# RUN groupadd -g $GID -o $UNAME
# RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
# RUN chown -cR /anime_gan_dir $USER
# USER $UNAME
## Build command start like that: docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) 

RUN pip install tensorflow-gpu==1.15.0 \
    opencv-python \
    tqdm \
    numpy \
    argparse \
    onnxruntime

# Downgrade protobuf - see https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
RUN pip install protobuf==3.20.*

COPY . .

ENTRYPOINT ["/bin/bash"]