# Docker Command List

## Basic Commands
```bash
# 1. Create a docker image from a Dockerfile
docker build -t my-image-name .

# 2. Create a docker container from a docker image
docker run -d --name my-container-name my-image-name

# 3a. List docker images
docker images

# 3b. List docker containers (running)
docker ps

# 3c. List docker containers (all)
docker ps -a

# 4a. Rerun a stopped docker container
docker start my-container-name

# 4b. Attach to a running docker container
docker attach my-container-name

# 5a. Rename a docker image
docker tag my-image-name my-new-image-name

# 5b. Rename a docker container
docker rename my-container-name my-new-container-name

# 5c. Delete a docker image
docker rmi my-image-name

# 5d. Delete a docker container
docker rm my-container-name

# Additional commands

# Stop a running container
docker stop my-container-name

# Access the shell of a running container
docker exec -it my-container-name /bin/bash

# Show logs of a container
docker logs my-container-name

# Save a container as a new image
docker commit CONTAINER_ID_OR_NAME new_image_name:tag

#save image as a tar file
docker save -o saved_image.tar new_image_name:tag

#load a tar file to image
docker load -i saved_image.tar

#reuse a stopped container that resume previous session, for continuing use GPU and GUI
docker start -ai my_dev_container

#configure the host machine
xhost +local:root

#run docker without sudo, run this and then log out
sudo usermod -aG docker $USER

docker run -it --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix 
ovmm_baseline:latest

# put the network in docker same as host machine
docker run -it --network="host" \
--gpus all -e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
z1_dev:v2
```

