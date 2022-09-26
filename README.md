# 
## First steps
To start this example you should first create a virtual env. 
```sh
$python3 -m venv ./venv_example
$source ./venv_example/bin/activate                        (Linux)
$.\venv_example\Scripts\activate.bat                 (Windows cmd)
```
After the venv has been created you should install the requirements
```sh
$pip install -r requirements.txt
$pip install -e .
```
The script model_training can be used to train and serialise a scikit-learn model
```sh
$python ./model/model_training.py
$git add ./model.pkl
$git commit -m 'aggiunto modello'
$git push
```

## Definition of a unit test
The file *test/test_data_and_model.py* contains a pytest example. Local tests can be run by using the command
```sh
$python -m pytest
```
At each execution pytest collects and runs every function containing the word 'test' in its name.

## Performing a grid-search with mlflow
The script run_grid_search contains the logic to perform a grid search using mlflow to store the results 
```sh
$python experiments/run_grid_search.py
```
You can access the mlflow user interface by running
```sh
$mlflow ui
```
You can use the query bar to search for specific experiments, for example
```
metrics.precision > 0.6 and params.depth='3'
```
