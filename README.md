# ML

Machine learning tools applied to datasets for prediction and classification. I added mathematical explanations wherever I could.

## References

- [Hands-On Machine Learning with Scikit-Learn and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781491962282/)
- [Dive into Deep Learning](https://d2l.ai/) 

## Dependencies

`scikit-learn`, `pytorch`, `tensorflow`, `numpy`, `pandas`, `matplotlib`

## Analysis of datasets 


* [`Housing.ipynb` ](Housing.ipynb): Predicting housing price in California using linear regression
- [`kaggle-housing.py`](torch/kaggle-housing.py) Predicting housing price in Ames (kaggle dataset) using softmax repression
* [`MNIST.ipynb`](MNIST.ipynb): Exploring MNIST data and classifying handwritten digits
- [`fashion_softmax_ready.py`](torch/fashion_softmax_ready.py): Classifying fashion-MNIST data using a single-layer neural network
- [`fashion_softmax.py`](torch/fashion_softmax.py) Softmax regression to classify fashion data implemented from scratch 
* [`logistic.py`](logistic.py): Classification of iris flowers using petal information using logistic regression
* [`lightdata.ipynb`](lightdata.ipynb): Learning the optimum brightness level of laptop screen based on ambient light and content of the screen
* [`happiness.py`](happiness.py): Predicting happiness quotients for different countries 
* [`logistic-money.py`](logistic-money.py): Logistic regression on expenditure to identify the user responsible for transaction


## Notes

- [torch](torch):  Implementation of neural networks in `pytorch`
- [`ml-notes.tex`](ml-notes.tex): My notes of linear, logistic, and softmax regression
- [`Models.ipynb`](Models.ipynb): Implementation of various models
- [`Support Vector Machine.ipynb`](svm.ipynb): Support vector machine
- [signal](signal.ipynb) : Signal processing on time series data using `scipy`

## Datasets

- [Happiness data by contry](datasets/BLI_20012019062939110.csv)
- [Housing price in California](datasets/housing.csv)
- [Housing price in Ames, IA](torch/data/kaggle-housing/house_tiny.csv)
- [Iris flowers](datasets/iris.data)
- [Laptop screen brightness data](datasets/lightdata)
- [Expensiture dataset](datasets/usable.csv)
- [GDP of countries](datasets/WEO_Data.xls)
