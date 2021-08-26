# Bioinformatics
Bioinformatics stuff that I am working on


## Feature Selection

### rf.features.py

This is a simple script to extract important features from a dataset via entropy. This script creates multiple decision trees with entropy gain formula. In a project, using this algorithm, I have extracted most important 20 genes from a dataset containing about 60.000 thousand genes. Using those 20 genes commonly used classfication algorithms could identify each sample up to '95 >' accuracy. Any number of labels and features can be used with this script. 

Output of the script is a csv file containing features and their total number of occurrences within all created trees.


To run the script, clone the repo with git, or just download it directly. 
```
git clone https://github.com/kcanakdag/Bioinformatics
```

Install the required packages with pip:

```
pip3 install -r requirements.txt
```

After installing the required packages you can test the script by running the following command;
```
python rf_features.py -h
```


#### Parameters

-i is a str, the input parameter, it is required and should be a path to a .csv dataset.

-min is an int, the amount of minimum samples required to classify a dataset in decision tree. Lower values may lead to overfitting. Default is 10.

-nf is an int, Number of features randomly selected with each decision tree training. So if you have a lot of features, randomly selecting a number of them shortens the time required to create decision trees greatly. Default is 100

-tt is an int, the train-test split size in percentage. Default is 20.

-n is an int, number of trees generated. default is 20

-acc is an int as accuracy cutoff for each tree in percentage. default is 90

-o is a str, the name of the output file-path.


#### Example usage with iris dataset;
```
python ./FeatureSelection/RandomForest/rf_feature.py -i ./test_datasets/iris.csv -n 500 -o ./rf_output.csv
```
