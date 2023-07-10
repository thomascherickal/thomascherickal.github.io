import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    SGDClassifier,
    Perceptron,
    LogisticRegressionCV,
    RidgeClassifierCV,
)
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.semi_supervised import (
    LabelPropagation,
    LabelSpreading,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multiclass import (
    OneVsOneClassifier,
    OneVsRestClassifier,
)
from sklearn.metrics import (
    balanced_accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    hinge_loss,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    multilabel_confusion_matrix,
    ndcg_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    zero_one_loss,
    silhouette_score,
)
from sklearn.ensemble import HistGradientBoostingClassifier

# Load the Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Preprocess the data
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
df.dropna(inplace=True)
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])

# Separate features and target variable
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a list of classifiers
classifiers = [
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    BaggingClassifier(),
    ExtraTreesClassifier(),
    GradientBoostingClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    PassiveAggressiveClassifier(),
    RidgeClassifier(),
    SGDClassifier(),
    Perceptron(),
    LogisticRegressionCV(),
    RidgeClassifierCV(),
    BernoulliNB(),
    CategoricalNB(),
    ComplementNB(),
    GaussianNB(),
    MultinomialNB(),
    KNeighborsClassifier(),
    RadiusNeighborsClassifier(),
    MLPClassifier(),
    SVC(),
    NuSVC(),
    LinearSVC(),
    GaussianProcessClassifier(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    IsotonicRegression(),
    MultiOutputClassifier(DecisionTreeClassifier()),
    MultiOutputClassifier(RandomForestClassifier()),
    MultiOutputClassifier(ExtraTreesClassifier()),
    MultiOutputClassifier(GradientBoostingClassifier()),
    OneVsOneClassifier(LinearSVC()),
    OneVsOneClassifier(LogisticRegression()),
    OneVsRestClassifier(LinearSVC()),
    OneVsRestClassifier(LogisticRegression()),
    LabelPropagation(),
    LabelSpreading(),
    CalibratedClassifierCV(),
    HistGradientBoostingClassifier(),
]

# Create a dictionary to store the accuracy scores
accuracy_scores = {}

X_train= X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

# Iterate over each classifier
for classifier in classifiers:
    # Fit the classifier on the training data
    classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = classifier.predict(X_test)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Store the accuracy score in the dictionary
    classifier_name = type(classifier).__name__
    accuracy_scores[classifier_name] = accuracy

# Create a DataFrame to display the accuracy scores
accuracy_df = pd.DataFrame.from_dict(accuracy_scores, orient='index', columns=['Accuracy'])
accuracy_df.index.name = 'Classifier'
accuracy_df.sort_values(by='Accuracy', ascending=False, inplace=True)

# Display the accuracy scores in a tabular form
print(accuracy_df)