from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# Load dataset
data = load_breast_cancer()
# Organize our data
label_names = data['target_names']  # vecteur des noms de classes
labels = data['target']
feature_names = data['feature_names']
features = data['data']
#Split our data
train, test, train_labels, test_labels = train_test_split(features,
             test_size=0.33, random_state=42)
# Initialize our classifier
gnb = GaussianNB()
Train our classifier
model = gnb.fit(train, train_labels)
# Make predictions
preds = gnb.predict(test)
# Evaluate accuracy
print(accuracy_score(test_labels, preds))