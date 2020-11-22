# Malaysia Car Plate Recognition System
This is a project on how to recognize malaysia car plate number using Deep Learning.

# Requirements
1. Python 3.6.10
```
$ conda create --name car-plate python=3.6.10
```
2. Install All the requirements
```
$ pip install -r requirements.txt
```

# Command to run
```
$ python main.py
```

# Docker Container
1. Build the Dockerfile
```
$ docker build -t car-plate:latest .
```

2. Run the Images
```
$ docker run -p 8000:8000 car-plate:latest -name carplate-container
```

3. SSH into docker container
```
$ docker start [docker id of the car plate container] (optional)
$ docker exec -it carplate-container /bin/bash
```

# API usage
```
The requirements.txt is updated, please install again
```
1.Run the server, it takes few minutes to set up
```
cd to src folder
python server.py
```
2. Open the interface by interface.html through any browser(double click it)
