import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error,r2_score
from tensorflow.keras.callbacks import EarlyStopping

# Enflasyon oranlarını bir liste olarak tanımla 
inflation_rates = {
    2010: 0.03, # 2010 için %3 enflasyon oranı
    2011: 0.05,
    2012: 0.02,
    2013: 0.04,
    2014: 0.01,
    2015: 0.03,
    2016: 0.02,
    2017: 0.03,
    2018: 0.04,
    2019: 0.02,
    2020: 0.01,
    2021: 0.03,
    2022: 0.04,
    2023: 0.02,
}

# Veriyi Excel dosyasından oku
dataFrame = pd.read_excel("merc.xlsx")

print(dataFrame.info())
print(dataFrame.describe())

# Eksik değerler varsa kontrol et
print(dataFrame.isnull().sum())

# Enflasyon oranlarına göre fiyatları ayarla
def adjust_for_inflation(price, year):
    rate = inflation_rates.get(year, 0)  # Verilen yıl için enflasyon oranını al
    adjusted_price = price * (1 + rate)  # Fiyatı enflasyon oranı ile ayarla
    return adjusted_price

dataFrame['adjusted_price'] = dataFrame.apply(lambda row: adjust_for_inflation(row['price'], row['year']), axis=1)

# Fiyatların dağılımını histogram ile göster
sbn.displot(dataFrame["adjusted_price"], kde=True)
plt.title('Enflasyona Göre Ayarlanmış Fiyat Dağılımı')
plt.xlabel('Fiyat')
plt.ylabel('Sıklık')
plt.show()

# Yıl bazında araç sayısını countplot ile göster
sbn.countplot(x="year", data=dataFrame)
plt.title('Yıllara Göre Araç Sayısı')
plt.xlabel('Yıl')
plt.ylabel('Araç Sayısı')
plt.xticks(rotation=45)
plt.show()

# Kilometre ve fiyat arasındaki ilişkiyi scatter plot ile göster
sbn.scatterplot(x="mileage", y="adjusted_price", data=dataFrame)
plt.title('Kilometre ve Ayarlanmış Fiyat İlişkisi')
plt.xlabel('Kilometre')
plt.ylabel('Ayarlanmış Fiyat')
plt.show()

# Kategorik sütunları sayısal verilere dönüştür
dataFrame = pd.get_dummies(dataFrame, columns=["transmission"], drop_first=True)

# Özellikler ve hedef değişkenleri ayır
y = dataFrame["adjusted_price"].values
x = dataFrame.drop(["price", "adjusted_price"], axis=1).values

# Özellikler ve hedef değişkeni eğitim ve test setlerine ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

# Özellikleri Min-Max ölçekleyici ile normalize et
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Yapay sinir ağı modelini oluştur
model = Sequential()
model.add(Dense(80, activation="relu", input_shape=(x_train.shape[1],)))
model.add(Dense(80, activation="relu"))
model.add(Dense(80, activation="relu"))
model.add(Dense(80, activation="relu"))
model.add(Dense(1))

# EarlyStopping callback'i tanımla
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Modeli derle ve öğrenme sürecini başlat
model.compile(optimizer="adam", loss="mse")
history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=250, epochs=300)

# Modelin öğrenme sürecindeki kayıp değerlerini göster
kayipVerisi = pd.DataFrame(history.history)
print(kayipVerisi.head())

# Kayıp değerlerinin grafiklerini çiz
plt.figure(figsize=(10, 5))
kayipVerisi.plot()
plt.title('Model Kayıp Değerleri')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend(['Eğitim', 'Doğrulama'])
plt.show()

# Test seti ile tahmin yap ve hata metriklerini hesapla
tahminDizisi = model.predict(x_test)
print(tahminDizisi)

# Ortalama mutlak hata değerini hesapla
print("Mean Absolute Error:", mean_absolute_error(y_test, tahminDizisi))

# R² skorunu hesapla
r2 = r2_score(y_test, tahminDizisi)
print("R-squared:", r2)

# Gerçek ve tahmin edilen fiyatları karşılaştır
plt.figure(figsize=(10, 5))
plt.scatter(y_test, tahminDizisi, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
plt.title('Gerçek ve Tahmin Edilen Ayarlanmış Fiyatlar')
plt.xlabel('Gerçek Fiyatlar')
plt.ylabel('Tahmin Edilen Fiyatlar')
plt.show()

# 22. satırdaki verileri al
print(dataFrame.iloc[22])

# 22. satırdaki verileri bir pandas serisi olarak al
yeniArabaSeries = dataFrame.drop(["price", "adjusted_price"], axis=1).iloc[22]

# Seriyi NumPy dizisine çevir ve ölçekleyici ile dönüştür
yeniArabaSeries = scaler.transform(yeniArabaSeries.values.reshape(1, -1))

# Bu yeni veriyi model ile tahmin et
predicted_price = model.predict(yeniArabaSeries)
print("Predicted price:", predicted_price[0][0])
print("Actual price:", dataFrame.iloc[22]["adjusted_price"])

# Kullanıcıdan araç bilgilerini al
def get_user_input():
    year = int(input("Yıl: "))
    mileage = float(input("Kilometre: "))
    tax = float(input("Vergi: "))
    mpg = float(input("MPG: "))
    engineSize = float(input("Motor Hacmi: "))
    transmission = input("Vites (Automatic veya Manual): ")

    transmission_automatic = 1 if transmission.lower() == "automatic" else 0

    user_data = {
        "year": year,
        "mileage": mileage,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engineSize,
        "transmission_Automatic": transmission_automatic
    }

    return user_data

user_data = get_user_input()
user_data_df = pd.DataFrame([user_data])
user_data_df = user_data_df.reindex(columns=dataFrame.columns.drop(["price", "adjusted_price"]), fill_value=0)

# Kullanıcı verisini ölçekle
user_data_scaled = scaler.transform(user_data_df)

# Fiyat tahmini yap
predicted_price = model.predict(user_data_scaled)
print("Tahmin Edilen Ayarlanmış Fiyat:", predicted_price[0][0])

