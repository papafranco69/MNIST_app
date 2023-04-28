from sklearn.metrics import precision_score, recall_score, roc_curve, f1_score, auc
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np

class Preprocessing:
    """
    Preprocesses the MNIST dataset
    applying feature scaling using StandardScaler 
    and dimensionality reduction using PCA.

    Parameter:
    ----------
    X: pd.Dataframe training features

    Return:
    np.array of the preprocessed data

    Example:
    ----------
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from utils import Preprocessing

    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

    X, y = mnist['data'], mnist['target']
    y = y.astype(np.uint8)

    preprocessor = Preprocessing(X)
    X_pca = preprocessor.fit_transform()


    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    
    """
    def __init__(self, X):
        self.X = X
    def fit_transform(self):
        preprocessor = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=50))
        ])
        X_pca = preprocessor.fit_transform(self.X)
        return X_pca




class Metrics:
    """
    Get evalution scores.

    parameters:
    ------------
    y_test: pd.Serie True labels of the test data
    y_pred: pd.Series The predicted labeles of the test data
    average: default is "macro" for multiclass target


    Examples:
    ------------
    from utils import Metrics

    knn = KNN_scratch(k=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    evaluation = Metrics(y_test, y_pred)
    precision = evaluation.get_precision()

    """
    def __init__(self, y_test, y_pred, average="macro"):
        self.y_test = y_test
        self.y_pred = y_pred
        self.average = average
    
    def get_precision(self):
        """
        Precision: is TP/ TP + FP, the accuracy of the postive predictions.
        That is when the model claims a sample belongs to a certain class,
        __%(precision) of the time it's correct.

        Return:
        float: weighted average of the precision of
        each class for the multiclass task
        """
        return precision_score(self.y_test, self.y_pred, average=self.average)
    
    def get_recall(self):
        """
        Recall(True Positive Rate): TP / TP + FN
        Recall is the ratio of the positive instances that
        are correctly detected by the classifier.

        Return:
        float: weighted average of the recall of each 
        class for the multiclass task

        """

        return recall_score(self.y_test, self.y_pred, average=self.average)
    
    def get_f1_score(self):
        """
        F1 Score: The F1 Score is the harmonic mean of precision and recall

        Return:
        float: weighted average of the f1 score of 
        each class for the multiclass task

        """

        return f1_score(self.y_test, self.y_pred, average=self.average)

class plot_ROC:
    """
    The ROC curve
    The dotted line represents the ROC curve of a purely random classifier;
    a good classifier stays as far away from that line as possible (toward the top-left corner).

    Area Under the Curve (AUC or just area):
    A perfect classifier will have a AUC equal to 1, 
    whereas a purely random classifier will have a AUC equal to 0.5.

    Using label_binarize function to transform the target variables into binary form,
    then computing the ROC curve and the area for each class separately.
    Comupting micro-average ROC cureve and area with the binary form of the target variables

    Parameters:
    -----------
    model: the machine learning model that has already been trained on the training data
    X_test: pd.Dataframe test data features
    y_test: pd.Serie True labels of the test data
    n_class: int the number of classes in the target variable, default 10 for the MNIST dataset

    """

    def __init__(self, model, X_test, y_test, n_class=10):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.n_class = n_class

    def plot(self):
        """
        generates the ROC curve for the model on the test data,
        and returns the FPR, TPR, and AUC scores for each class

        Example:
        from utils import plot_ROC

        knn.fit(X_train, y_train)
        plot = plot_ROC(knn, X_test, y_test, n_class=10)
        plot.plot()

        """

        y_score = self.model.predict_proba(self.X_test)
        y_test = label_binarize(self.y_test, classes=np.arange(self.n_class))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(y_test.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot ROC curves
        plt.figure()
        lw = 2
        plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
                lw=lw, label='micro-average ROC curve (area = %0.2f)' % roc_auc["micro"])
        for i in range(y_test.shape[1]):
            plt.plot(fpr[i], tpr[i], lw=lw,
                    label='ROC curve of class %d (area = %0.2f)' % (i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for KNN_scratch on MNIST')
        plt.legend(loc="lower right")
        plt.show()

