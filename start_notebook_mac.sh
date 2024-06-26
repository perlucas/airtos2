# !bin/sh

podman run \
-it \
-p 8888:8888 \
-p 6006:6006 \
--user root \
-e GRANT_SUDO=yes \
-v "/Users/lucaspereyra/luqui/airtos2/notebooks":/home/jovyan/work \
jupyter/tensorflow-notebook

# podman run \
# -it \
# -p 8888:8888 \
# --rm \
# -v "/Users/lucaspereyra/luqui/airtos2/notebooks":/home/jovyan/work \
# jupyter/datascience-notebook 