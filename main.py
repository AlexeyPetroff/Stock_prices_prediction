import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []


def get_data(filename):
    with open(filename, 'r') as csvfile:
        csv_file_reader = csv.reader(csvfile)
        next(csv_file_reader)
        t=0
        for row in csv_file_reader:
            dates.append(t)
            prices.append(float(row[1]))
            t+=1


def predict_prices(dates, prices, x):
    dates = dates[::-1]
    dates = np.reshape(dates, (len(dates), 1))

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black')
    plt.plot(dates, svr_rbf.predict(dates), color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.show()

    return svr_rbf.predict(x)[0]


get_data('wig20.csv')

predicted_price = predict_prices(dates, prices, 29)
print(predicted_price)