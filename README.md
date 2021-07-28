# currency_nn
Neural net project with the kaggle currency dataset: https://www.kaggle.com/brunotly/foreign-exchange-rates-per-dollar-20002019

Dataset is augmented with EMU convergence criterion series data: https://data.europa.eu/data/datasets/o9zmuyfsqergpapzcuzsqg?locale=en

LSTM notebook is based off of code from this Google Colab notebook: https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/pytorch/Time_Series_Prediction_with_LSTM_Using_PyTorch.ipynb#scrollTo=CKEzO1jzKydL



## Notebooks

* Regression Exploration: Fully connected regression model (no LSTM). Achieves excellent R^2 error. We don't spend any time on regression outside of this

* Base Classification: Fully connected classification model (no LSTM). Highlights the data used and the transformation functions

* LSTM: Experiments with a an LSTM with a single fully connected layer at the end.

* DeepNet: Experiment with an LSTM with multiple fully connected layers at the end.

* Backtesting: Framework for backtesting models