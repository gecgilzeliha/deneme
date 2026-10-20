import pandas as pd
import string

# 1. Veriyi okuma
df = pd.read_csv("starbucks_customer_ordering_patterns.csv")

# 2. Sadece işimize yarayacak sütunları (Yaş ve İçecek) seçelim 
# ve içi boş (NaN) olan hatalı satırları atalım
df = df[['customer_age_group', 'drink_category']].dropna()

# 3. METİN TEMİZLEME (SÜZME)
# Önce tüm harfleri küçültelim
df['temiz_metin'] = df['drink_category'].str.lower()

# Noktalama işaretlerini temizleyen küçük bir fonksiyon yazalım
def noktalama_temizle(metin):
    # Eğer metin string formatındaysa noktalama işaretlerini sil
    return str(metin).translate(str.maketrans('', '', string.punctuation))

# Bu fonksiyonu sütuna uygulayalım
df['temiz_metin'] = df['temiz_metin'].apply(noktalama_temizle)

# Temizlenmiş verinin ilk 10 satırına bakalım
print(df[['customer_age_group', 'drink_category', 'temiz_metin']].head(10))

from sklearn.feature_extraction.text import CountVectorizer

print("\n--- İkili Kelime (Bi-gram) Analizi ---")

# ngram_range=(2,2) parametresi kelimeleri tek tek değil, peş peşe gelen ikili gruplar halinde almamızı sağlar
vectorizer = CountVectorizer(ngram_range=(2, 2))

# Sadece temizlenmiş metinlerimizi analiz aracına veriyoruz
X = vectorizer.fit_transform(df['temiz_metin'])

# Hangi kelime çiftinin kaç kez geçtiğini topluyoruz
ikili_kelimeler = vectorizer.get_feature_names_out()
frekanslar = X.sum(axis=0).A1 # Matrisi tek boyutlu diziye çevirir

# Kelime çiftlerini ve frekanslarını birleştirip Pandas DataFrame'e koyalım ki kolay sıralansın
bigram_df = pd.DataFrame({'Kelime_Ikilisi': ikili_kelimeler, 'Frekans': frekanslar})
bigram_df = bigram_df.sort_values(by='Frekans', ascending=False).reset_index(drop=True)

# En çok yan yana gelen ilk 10 kelime çiftine bakalım
print("En Çok Yan Yana Gelen Kelimeler:")
print(bigram_df.head(10))

import matplotlib.pyplot as plt

print("\n--- Yaş Gruplarına Göre Dağılım ---")

# Yaş grubuna ve içecek metnine göre gruplayıp sipariş sayılarını (frekans) hesaplıyoruz
yas_icecek_dagilimi = df.groupby(['customer_age_group', 'drink_category']).size().unstack()

# Veriyi bir sütun grafiği (bar chart) olarak çizdiriyoruz
yas_icecek_dagilimi.plot(kind='bar', figsize=(10, 6), colormap='viridis')

# Grafiğin başlık ve eksen isimlerini ekliyoruz
plt.title("Yaş Gruplarına Göre En Çok Aranan/Sipariş Edilen Ürünler")
plt.xlabel("Yaş Grupları")
plt.ylabel("Sipariş Sayısı (Frekans)")
plt.xticks(rotation=45) # Yaş etiketleri birbirine girmesin diye 45 derece eğiyoruz
plt.legend(title="İçecek Kelimesi", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Grafiği ekranda gösteriyoruz
plt.show()