import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv("phosphate.csv")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
table= pd.pivot_table(data, values='Water Usage', index=['Ore'],columns=['Sulphur','Phosphoric','ammonia'], aggfunc=np.mean)
ax = table.T.plot(kind='bar')
plt.title("Phosphate Minerals Data")
plt.xlabel("Chemicals")
plt.ylabel("Water Usage")
plt.show()
def algorithm(datas):
    # print(datas)
    # data = pd.DataFrame(pd.read_excel("waste dataset.xlsx"))
    # read_file = pd.read_excel("waste dataset.xlsx")
    # read_file.to_csv("waste dataset.csv", header=True, index=False)
    data = pd.DataFrame(pd.read_csv("waste dataset.csv"))
    # data = pd.read_csv('Construction Cost.csv')
    data_x = data.iloc[:, :-1]
    data_y = data.iloc[:, -1]
    string_datas = [i for i in data_x.columns if data_x.dtypes[i] == np.object_]

    LabelEncoders = []
    for i in string_datas:
        newLabelEncoder = LabelEncoder()
        data_x[i] = newLabelEncoder.fit_transform(data_x[i])
        LabelEncoders.append(newLabelEncoder)
    ylabel_encoder = None
    if type(data_y.iloc[1]) == str:
        ylabel_encoder = LabelEncoder()
        data_y = ylabel_encoder.fit_transform(data_y)

    model = Ridge()
    model.fit(data_x, data_y)
    value = {data_x.columns[i]: datas[i] for i in range(len(datas))}
    l = 0
    for i in string_datas:
        z = LabelEncoders[l]
        value[i] = z.transform([value[i]])[0]
        l += 1
    value = [i for i in value.values()]
    predicted = model.predict([value])
    if ylabel_encoder:
        predicted = ylabel_encoder.inverse_transform(predicted)
    return predicted[0]


a = algorithm(['Organic_waste','Agricultural_Waste','Solid_Waste'])
print(a)
