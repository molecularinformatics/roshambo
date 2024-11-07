# Put your data in the data/test folder or change the path of the volume. 
# The output will be in the same folder.
# Make sure your output folder is created before running the docking environement 
# otherwise it will be created and lead to an error of permission on save.


mkdir ./data/test
docker run --rm --gpus all --user $(id -u):$(id -g) -v ./data/test:/home/appuser/roshambo/data/volume roshambo --n_confs 10 --ignore_hs --color --sort_by ComboTanimoto --write_to_file --working_dir /home/appuser/roshambo --align_confs --num_threads 8 --gpu_id 0 --filename data/volume/hits.sdf notebooks/data/grid/query.sdf notebooks/data/grid/dataset.sdf