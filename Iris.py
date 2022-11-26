# Модуль Lerning_Iris
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Загрузка из Интернета массива из 150 элементов
# Загрузка их в объект DataFrame и печать
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
print('Массив')
print(df.to_string())
# Выборка из объекта DF 100 элементов (столбец 4 - название цветков),
# загрузка его в одномерный массив У и печать
y= df.iloc[0:100, 4].values
print ('Значение четвертого столбца y - 100')
print(y)
# Преобразование названий цветков (столбец 4)
# в одномерный массив (вектор) из -1 и 1
y = np.where(y== 'Iris-setosa', -1, 1)
print('Значение названий цветков в виде -1 и 1, У - 100')
print(y)
# Выборка из объекта DF массива 100 элементов (столбец О и столбец 2),
# загрузка его в массив Х (матрица) и печать
x = df.iloc[0: 100, [0, 2]].values
print('Значение Х - 100')
print(x)
print('Конец Х')
# Формирование параметров значений для вывода на график
# Первые 50 элементов (строки 0-50, столбцы О, 1)
plt.scatter(x[0:50, 0], x[0:50, 1], color='red', marker='o', lаЬеl='щетинистый')
# Следующие 50 элементов (строки 50-100, столбцы О, 1)
plt.scatter(x[50:100, 0], x[50:100, 1], color='Ьlue', marker='x', lаbеl='разноцветный')
# Формирование названий осей и вывод графика на экран
plt.xlabel('Длина чашелистика')
plt.ylabel('Длина лепестка')
plt.legend(loc='upper left')
plt.show()
# Описание класса Perceptron
class Perceptron(object):
    '''
    Классификатор на основе персептрона.
    Параметры
    eta:float - Темп обучения (междуО.О и 1.0)
    п iter:int - Проходы по тренировочному наборуданных.
    Атрибуты
    w_: 1-мерный массив - Весовые коэффициенты после подгонки.
    errors : список - Число случаев ошибочной классификации в каждой эпохе.
    '''
    def __init__ (self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter= n_iter
    '''
    Быполнить подгонку модели под тренировочные данные.
    Параметры
    Х : массив, форма = (n_sam ples, n_features] тренировочные векторы,
    где
    n_saпples - число образцов и
    п features - число признаков,
    (п_saпples] Целевые значения.
    у: массив, форма
    Возвращает
    self: object
    '''
    def fit(self, х, у):
        self.w=np.zeros(1 + x.shape[1])
        self.errors = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, у):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                self.errors_.append(errors)
        return self
#Рассчитать чистый вход 
def net_input(self, Х):
return np.dot(X, self.w_[l:]) + self.w_[O]
Вернуть меткукласса после единичного скачка 111
def predict(self, Х):
return np.where(self.net_input(X) >= О.О, 1, -1)
# Тренировка
ppn = Perceptron(eta�O.l, n iter= lO)
ppn.fit(X, у)
plt.plot(range(l, len(ppn.errors ) + 1), ppn.errors_, marker='o')
plt.xlabel('Эпохи')
# число ошибочно классифицированных случаев во время обновлений
plt.ylabel('Чиcлo ·случаев ошибочной классификации')
plt.show()
# Визуализация границы решений
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, у, classifier, resolution= 0.02):
# настроить генератор маркеров и палитру
markers = ('s', 'х', 'о', 'л', 'v')
colors = ('red', 'Ьlue', 'green', 'gray', 'cyan')
стар = ListedColormap(colors[:len(np.unique(y))])
# вывести поверхность решения
xl_min, xl_max = Х[:, О].min() - 1, Х[:, О].max() + 1
x2_min, x2_max = Х[:, 1].min() - 1, Х[:, 1].max() + 1
xxl, хх2 = np.meshgrid(np.arange(xl_min, xl_max, resolution),
np.arange(x2_min, x2_max, resolution))
Z = classifier.predict(np.array([xxl.ravel(), xx2.ravel()]).Т)
Z = Z.reshape(xxl.shape)
plt.contourf(xxl, хх2, Z, alpha=0.4, cmap=cmap)
plt.xlim(xxl.min(), xxl.max())
plt.ylim(xx2.min(), xx2.max())
# показать образцы классов
for idx, cl in enumerate(np.unique(y)):
plt.scatter(x=X[y == cl, О], у=Х[у == cl, 1], alpha=0.8,
c=cmap(idx), marker=markers[idx], label=cl)
# Нарисовать картинку
plot_decision_regions(X, у, classifier=ppn)
plt.xlabel('Дпина чашелистика, см')
plt'.уlabel('Дпина лепестка, см')
plt.legend(loc='upper left')
plt.show()
# Адаптивный линейный нейрон
class AdaptiveLinearNeuron(object):
Классификатор на основе ADALINE (ADAptive Linear NEuron).
Параметры
eta : floa t - Темп обучения (между О. О и 1 . О)
п iter: in - Проходы по тренировочному набору данных
Атрибуты
w : 1-мерный массив - Веса после подгонки.
errors
эпохе.
: список - Число случаев ошибочной классификации в каждой
'''
def
init (self, rate=0.01, niter=lO):
self.rate = rate
self.niter = niter
def fit(self, Х, у):
''' Быполнить подгонку под тренировочные данные. ПараметрыХ: (массив}, форма = [n_saпples, n_features) - тренировочные векторы,
где п_saпples .:. число образцов,
п features - число признаков.
у: массив, форма
[n_saпples) - Целевые значения.
Возвращает
self: объект
'''
self.weight = np.zeros(l + X.shape[l])
self.cost = []
for i in range(self.niter):
output = self.net_input(X)
errors = у - output
self.weight[l:] += self.rate * X.T.dot(errors)
self.weight[O] += self.rate * errors.sum()
cost = (errors**2).sum() / 2.0
self.cost.append(cost)
return self
def net_input(self, Х):
# Вычисление чистого входного сигнала
return np.dot(X, self.weight[l:]) + self.weight[OJ
def activation(self, Х):
# Вычислительная линейная активация
return self.net_input(X)
11
def predict(self, Х):
# Вернуть метку класса после единичного скачка
return np.where(self.activation(X) >= О.О, 1, -1)
fig, ах = plt.suЬplots(nrows= l, ncols =2, figsize=(8, 4))
# learning rate = 0.01
alnl = AdaptiveLinearNeuron(0.01, 10).fit(X,y)
ax[0].plot(range(l, len(alnl.cost) + 1), np.logl0(alnl.cost), marker='o')
ах[О].set_xlabel('Эпохи')
ах[О).set_ylabel('log(Сумма квадратичных ошибок)')
ах[О] .set_title('ADALINE. Темп обучения 0.01')
# learning rate = 0.01
aln2 = AdaptiveLinearNeuron(0.0001, 10).fit(X,y)
ax[l] .plot(range(l, len(aln2.cost) + 1), aln2.cost, marker='o')
ах[1].set_xlabel('Эпохи')
ax[l].set_ylabel('Cyммa квадратичных ошибок')
ах[1].set_title('ADALINE. Темп обучения О.0001')
plt.show()
X_std = np.copy(X)
X_std[:, О] (Х[:, О] - Х[:, 0].mean()) / Х[:, 0].std()
X_std[:, 1] = (Х[:, 1) - Х[:, 1].mean()) / Х[:, 1].std()
# learning rate = 0.01
aln = AdaptiveLinearNeuron(0.01, 10)
aln.fit(X_std, у)
# нарисовать картинку
plot_decision_regions(X_std, у, classifier= aln)
plt.title('ADALINE (градиентный спуск)')
рlt.хlаЬеl('Длина чашелистика [стандартизованная]')
рlt.уlаЬеl('Длина лепестка [стандартизованная]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(l, len(aln.cost) + 1), aln.cost, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Cyммa квадратичных ошибок')
plt.show()