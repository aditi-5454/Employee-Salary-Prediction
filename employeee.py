import pandas as pd

data = pd.read_csv(r'D:\EMPLOYEE\adult 3.csv.csv')

data

data.shape

data.isna()

data.isna().sum()

print(data.occupation.value_counts())

print(data.gender.value_counts())

print(data.education.value_counts())

print(data.workclass.value_counts())

print(data["marital-status"].value_counts())
data=data[data['marital-status']!='Married-AF-spouse']

data.occupation.replace({'?':'Others'},inplace=True)

print(data.occupation.value_counts())

data.workclass.replace({'?':'NotListed'},inplace=True)

print(data.workclass.value_counts())

data=data[data['workclass']!='Without-pay']
data=data[data['workclass']!='Never-worked']

print(data.workclass.value_counts())

print(data.education.value_counts())

data=data[data['education']!='Preschool']
data=data[data['education']!='1st-4th']
data=data[data['education']!='5th-6th']

print(data.relationship.value_counts())

#redundancy
data.drop(columns=['education'],inplace=True)

data.drop(columns=['relationship'],inplace=True)

data.drop(columns=['fnlwgt'],inplace=True)

data.head()

print(data.race.value_counts())

print(data["hours-per-week"].value_counts())

print(data["native-country"].value_counts())

data['native-country'].replace(' ?', 'Unknown', inplace=True)

import matplotlib.pyplot as plt
plt.boxplot(data['age'])
plt.show()

data=data[(data['age']<=75) & (data['age']>=17)]

plt.boxplot(data['age'])
plt.show()

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['workclass']=encoder.fit_transform(data['workclass'])
data['marital-status']=encoder.fit_transform(data['marital-status'])
data['occupation']=encoder.fit_transform(data['occupation'])
data['race']=encoder.fit_transform(data['race'])
data['gender']=encoder.fit_transform(data['gender'])
data['native-country']=encoder.fit_transform(data['native-country'])

X=data.drop(columns=['income'])
Y=data['income']

X
Y

data

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X=scaler.fit_transform(X)
X

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=23,stratify=Y)

xtrain


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    
}

results = {}

for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    
model_names = list(results.keys())
accuracies = list(results.values())

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, accuracies, color='skyblue')
plt.ylim(0, 1.0)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

# Add accuracy values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
}

results = {}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# Get best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# Save the best model
joblib.dump(best_model, "best_model.pkl")
print(" Saved best model as best_model.pkl")