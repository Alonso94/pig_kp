# PIG   
Prepare the environment:<br>
```
sh prepare_env.sh
```
Download videos from MIME data set by listing their numbers [1-20]:<br>
```
/bin/bash download_mime.sh 1 2 
```
Download videos from SIMITATE data set by listing their numbers [1-4]:<br>
```
/bin/bash download_simitate.sh 1 2 
```
Activate the conda environment and run a test<br>
```
conda activate pig
python scripts/test.py
```