import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping

# Veriyi Excel dosyasından oku
dataFrame = pd.read_excel("merc.xlsx")

print(dataFrame.info(5))
print(dataFrame.describe())

# Eksik değerler varsa kontrol et
print(dataFrame.isnull().sum())

# Fiyatların dağılımını histogram ile göster
sbn.displot(dataFrame["price"], kde=True)
plt.title('Fiyat Dağılımı')
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
sbn.scatterplot(x="mileage", y="price", data=dataFrame)
plt.title('Kilometre ve Fiyat İlişkisi')
plt.xlabel('Kilometre')
plt.ylabel('Fiyat')
plt.show()

# Özelliklerin korelasyon analizini yap ve görselleştir
plt.figure(figsize=(12, 8))
sbn.heatmap(dataFrame.corr(), annot=True, cmap="coolwarm")
plt.title('Özelliklerin Korelasyon Matrisi')
plt.show()

# Kategorik sütunları sayısal verilere dönüştür
dataFrame = pd.get_dummies(dataFrame, columns=["transmission"], drop_first=True)

# Özellikler ve hedef değişkenleri ayır
y = dataFrame["price"].values
x = dataFrame.drop("price", axis=1).values

# Özellikler ve hedef değişkeni eğitim ve test setlerine ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

# Özellikleri Min-Max ölçekleyici ile normalize et
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Yapay sinir ağı modelini oluştur
model = Sequential()
model.add(Dense(12, activation="relu", input_shape=(x_train.shape[1],)))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
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
"""print("Mean Absolute Error:", mean_absolute_error(y_test, tahminDizisi))"""

# Gerçek ve tahmin edilen fiyatları karşılaştır
plt.figure(figsize=(10, 5))
plt.scatter(y_test, tahminDizisi, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
plt.title('Gerçek ve Tahmin Edilen Fiyatlar')
plt.xlabel('Gerçek Fiyatlar')
plt.ylabel('Tahmin Edilen Fiyatlar')
plt.show()



# 2. satırdaki verileri al
print(dataFrame.iloc[22])

# 2. satırdaki verileri bir pandas serisi olarak al
yeniArabaSeries = dataFrame.drop("price", axis=1).iloc[22]

# Seriyi NumPy dizisine çevir ve ölçekleyici ile dönüştür
yeniArabaSeries = scaler.transform(yeniArabaSeries.values.reshape(1, -1))

# Bu yeni veriyi model ile tahmin et
predicted_price = model.predict(yeniArabaSeries)
print("Predicted price:", predicted_price[0][0])
print("Actual price:", dataFrame.iloc[22]["price"])



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
user_data_df = user_data_df.reindex(columns=dataFrame.columns.drop("price"), fill_value=0)

# Kullanıcı verisini ölçekle
user_data_scaled = scaler.transform(user_data_df)

# Fiyat tahmini yap
predicted_price = model.predict(user_data_scaled)
print("Tahmin Edilen Fiyat:", predicted_price[0][0])






from sklearn.metrics import r2_score

# Test seti ile tahmin yap ve hata metriklerini hesapla
tahminDizisi = model.predict(x_test)

# Ortalama mutlak hata değerini hesapla
mae = mean_absolute_error(y_test, tahminDizisi)
print("Mean Absolute Error:", mae)

# R² skorunu hesapla
r2 = r2_score(y_test, tahminDizisi)
print("R-squared:", r2)


