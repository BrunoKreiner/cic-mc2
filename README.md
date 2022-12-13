# cic-mc2

Repository for the second minichallenge of the cic module. 
Download preprocessed data to data/processed from here: https://drive.google.com/file/d/1SFlUqgr5E07xhN5_qJLiJ1ZwRHoG098P/view?usp=sharing
Raw data was over 70gb and was preprocessed using "Preprocessing.py" using dask parallelization. The raw data is first splitted into train/test csv files and then normalized using the libraries torchio and monai. 
The data is then stored to data/processed. The train/test files are updated to match the new image location. The same file has a code block for the neural network training without dask.

In "CNN_on_cluster.py" is the code I used on a SaturnCloud cluster to train the deep learning model.
