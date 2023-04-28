# The MNIST Application

### Introduction

The MNIST Application provides a practical approach to learning machine learning models. It allows users to draw random digits ranging from 0 to 9, select a machine learning model, and then fine-tune the parameters. With no coding requirements, users can gain knowledge on how each parameter affects the outcomes. The application presents a ROC curve graph along with precision, recall, and f1 scores for evaluation metrics.

The application employs the MNIST dataset and offers users four model options, including two distinct implementations of the K-nearest Neighbors algorithm and Randomforest.

---------

### Sample Code

#### Original Implementation of the K-Nearest-Neighbors algorithm

~~~python
from models import KNN_scratch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

#Import the dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

X, y = mnist["data"], mnist["target"]
# cast y to integers
y = y.astype(np.uint8)


n_neighbor = 3
knn_scratch = KNN_scratch(k=n_neighbors)
knn_scratch.fit(X_train, y_train)
y_pred = knn_scratch.predict(X_test)

~~~

#### Optional data preprocessing function
Returns a NumPy array with the preprocessed data after applying feature scaling using StandardScaler and dimensionality reduction using PCA.

~~~python
from utils import Preprocessing
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

#Import the dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

X, y = mnist["data"], mnist["target"]
# cast y to integers
y = y.astype(np.uint8)


preprocessor = Preprocessing(X)
X_pca = preprocessor.fit_transform()


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)


~~~

#### Evaluation Metrics
Get precision, recall, and f1 score

~~~python
from utils import Metrics

knn = KNN_scratch(k=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

evaluation = Metrics(y_test, y_pred)
precision = evaluation.get_precision()


~~~

#### Plot ROC Curve
The dotted line represents the ROC curve of a purely random classifier;
a good classifier stays as far away from that line as possible (toward the top-left corner).

Area Under the Curve (AUC or just area):  
A perfect classifier will have a AUC equal to 1, 
whereas a purely random classifier will have a AUC equal to 0.5.

~~~python
from utils import plot_ROC

knn = KNN_scratch(k=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

plot = plot_ROC(knn, X_test, y_test)
plot.plot()

~~~


![Alt Image Text](https://github.com/papafranco69/MNIST_app/blob/main/image/Screenshot%202023-04-28%20at%2014.42.30.png)
