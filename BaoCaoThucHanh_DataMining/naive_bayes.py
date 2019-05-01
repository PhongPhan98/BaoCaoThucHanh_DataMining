from sklearn import metrics
import pandas as pd
weather_df = pd.read_csv('weather.csv')
#tiền xử lý dữ liệu
weather_dfdata=weather_df.drop(columns=['WindGustDir', 'WindDir9am','WindDir3pm'])
weather_dfdata.loc[weather_dfdata['RainTomorrow']=='No','RainToday']=0
weather_dfdata.loc[weather_dfdata['RainTomorrow']=='Yes','RainToday']=1
weather_dfdata.loc[weather_dfdata['RainTomorrow']=='No','RainTomorrow']=0
weather_dfdata.loc[weather_dfdata['RainTomorrow']=='Yes','RainTomorrow']=1
print(weather_dfdata)
weather_dfdata=weather_dfdata.fillna(method='ffill')
weather_dfdata=weather_dfdata.fillna(method ='backfill')
# test data
weather_dftest=weather_dfdata.tail(67)
data_test=weather_dftest.loc[:,"MinTemp":"RISK_MM"]
data_testy=weather_dftest['RainTomorrow']

weather_dfdata=weather_dfdata.head(300)
train_data=weather_dfdata.loc[:,"MinTemp":"RISK_MM"]
print(train_data)
label=weather_dfdata['RainTomorrow']
print(label)
from sklearn.naive_bayes import GaussianNB
## call MultinomialNB
clf = GaussianNB()
# training
clf.fit(train_data,label)
# test với một bộ dữ liệu có thể thay đổi
# bạn có thể thay đối số 2 với bất kì số nào trong khoảng 0 đến 67

d5=data_test.iloc[[2]]
print(d5)
print(clf.predict(d5)[0])
if str(clf.predict(d5)[0])=='0':
    print('No')
else:
    print('Yes')
y_predict=clf.predict(data_test)
#tính toán độ chính xác
print(metrics.accuracy_score(data_testy,y_predict))

