from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train,y_test = train_test_split(X,y,
                                                   train_size=0.875,test_size=0.125,random_state=188)

model = LogisticRegression(penalty="l2",C=1.0,random_state=None,solver="lbfgs", max_iter=3000,multi_class="ovr",verbose=1)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_pred_probs = model.predict_proba(X_test)

print('-'*10+'report'+'-'*10)
print(classification_report(y_test,y_pred))

print('-'*10+'params'+'-'*10)
print(model.coef_, model.intercept_)

print('-'*10+'auc'+'-'*10)
print(roc_auc_score(y_test,y_pred))

