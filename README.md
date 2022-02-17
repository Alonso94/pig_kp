# PIG   
Prepare the environment:<br>
```
sh prepare_env.sh
```
Download tasks from the MIME data set by listing their numbers [1-20]:<br>
```
sh download_mime.sh 1 2 
```
Activate the conda environment and run a test<br>
```
conda activate pig
python scripts/test.py
```