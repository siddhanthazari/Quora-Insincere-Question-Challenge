from ReadData import get_matrices, get_filtered_words, get_lem_dict
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score, f1_score
import numpy as np


x, y = get_matrices()
print("x shape = " + str(x.shape))
print("y shape = " + str(y.shape))

folds = 5
word_counts = 10
clf = MultinomialNB()

kf = KFold(n_splits=folds, shuffle=True)
kf.get_n_splits(x)

test_acc = []
test_mean = []
test_cm = np.array([[0, 0], [0, 0]])
test_f1 = []
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    test_acc.append(accuracy_score(y_test, y_pred))
    test_mean.append(balanced_accuracy_score(y_test, y_pred))
    test_cm += confusion_matrix(y_test, y_pred)
    test_f1.append(f1_score(y_test, y_pred))

print()
print("Test Accuracies using " + str(folds) + " folds = " + str(test_acc))
print("Test Mean Accuracies using " + str(folds) + " folds = " + str(test_mean))
print("Test F1 score using " + str(folds) + " folds = " + str(test_f1))
print("Test Confusion Matrix using" + str(folds) + " folds = \n" + str(test_cm))

print()
print("Test Average Accuracy = " + str(sum(test_acc) / len(test_acc)))
print("Test Average Mean Accuracy = " + str(sum(test_mean) / len(test_mean)))
print("Test Average F1 score = " + str(sum(test_f1) / len(test_f1)))
print("Test Normalized Confusion matrix = ")
print(test_cm / np.sum(test_cm, axis=1, keepdims=True))

clf.fit(x, y)

test = np.identity(x.shape[1])
prob = clf.predict_proba(test)[:, 1]
idx = np.argsort(-1 * prob)
vocab = get_filtered_words()
lem_dict = get_lem_dict()

print()
print("Words contributing highest to being insincere:")
for i in range(word_counts):
    print(str(i + 1) + ". word = " + str(vocab[idx[i]]) + ",\t probability = " + str(prob[idx[i]]) + ",\t usage = ", end ="")
    usage = [base_word for base_word, lem_word in lem_dict.items() if lem_word == vocab[idx[i]]]
    print(usage)

idx = np.argsort(prob)
vocab = get_filtered_words()
lem_dict = get_lem_dict()

print()
print("Words contributing least to being insincere:")
for i in range(word_counts):
    print(str(i + 1) + ". word = " + str(vocab[idx[i]]) + ",\t probability = " + str(prob[idx[i]]) + ",\t usage = ",
          end="")
    usage = [base_word for base_word, lem_word in lem_dict.items() if lem_word == vocab[idx[i]]]
    print(usage)
