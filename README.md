# Bioinformatics
Bioinformatics stuff that I am working on


## Feature Selection

### rf.features.py

This is a simple script to extract important features from a dataset via entropy. This script creates multiple decision trees with entropy gain formula. In a project, using this algorithm, I have extracted most important 20 genes from a dataset containing about 60.000 thousand genes. Using those 20 genes commonly used classfication algorithms could identify each sample up to '95 >' accuracy. Any number of labels and features can be used with this script.

To run the script, clone the repo with git, or just download it directly. 

Install the required packages with pip:

```
pip3 install -r requirements.txt
```

After installing the required packages you can run the following command;
```
python rf_features.py -h
```

Example usage with iris dataset;
```
python rf_feature.py -i ~/test_datasets/iris.csv -n 500 -o ./rf_output.csv
```



