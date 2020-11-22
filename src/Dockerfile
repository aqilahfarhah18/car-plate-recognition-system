FROM  centos:7

# copy our code to the docker
COPY . /application
WORKDIR /application

ENV LC_ALL en_US.utf8
ENV LANG en_US.utf8

RUN yum install -y python3-pip

# Requirements failes cause didn't update setuptools and pip to latest version
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools

RUN yum install -y python3-devel mysql-devel \
    gcc gcc-devel \
    libXext libSM libXrender \
    python-devel
RUN pip3 install -r requirements.txt

ENTRYPOINT python3 main.py

# EXPOSE 7000

# Need this to run commands everytime container is started
# ENTRYPOINT uvicorn serverFast:app --reload

