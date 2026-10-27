import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import string


st.set_page_config(page_title="Müşteri Analiz Paneli", layout="wide")

# Veriyi önbelleğe (cache) alalım ki sayfa her yenilendiğinde baştan okumasın (RAM dostu)
@st.cache_data
def load_data():
    df = pd.read_csv("starbucks_customer_ordering_patterns.csv")
    df = df[['customer_age_group', 'drink_category']].dropna()
    
    #  (Süzme)
    df['temiz_metin'] = df['drink_category'].str.lower()
    df['temiz_metin'] = df['temiz_metin'].apply(lambda x: str(x).translate(str.maketrans('', '', string.punctuation)))
    return df

df = load_data()

st.title("☕ Kahve Müşterisi Davranış ve Kelime Analizi")

# 1. SOL MENÜ (Sidebar) - Yaş Filtresi
st.sidebar.header("Filtreler")
yas_gruplari = sorted(df['customer_age_group'].unique())
secilen_yas = st.sidebar.selectbox("Yaş Grubu Seçin:", ["Tümü"] + yas_gruplari)

# Veriyi yaşa göre filtrele
if secilen_yas != "Tümü":
    df_filtrelenmis = df[df['customer_age_group'] == secilen_yas]
else:
    df_filtrelenmis = df

# 2. ANA EKRAN - Grafikler ve Analiz
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"{secilen_yas} Yaş Grubu En Çok Tercih Edilenler")
    # Seçilen yaş grubu için pasta grafiği (Pie Chart)
    icecek_sayilari = df_filtrelenmis['drink_category'].value_counts()
    
    fig1, ax1 = plt.subplots(figsize=(6,6))
    ax1.pie(icecek_sayilari, labels=icecek_sayilari.index, autopct='%1.1f%%', startangle=90, )
    ax1.axis('equal') 
    st.pyplot(fig1)

with col2:
    st.subheader("Kelime Bağlantı Analizi (Bi-gram)")
    st.write("Sistem, aranan kelimenin hemen arkasından gelen kelimeleri hesaplar.")
    
    # NLP Bi-gram Analizi
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    try:
        X = vectorizer.fit_transform(df_filtrelenmis['temiz_metin'])
        ikili_kelimeler = vectorizer.get_feature_names_out()
        frekanslar = X.sum(axis=0).A1
        
        bigram_df = pd.DataFrame({'Kelime İkilisi': ikili_kelimeler, 'Frekans': frekanslar})
        bigram_df = bigram_df.sort_values(by='Frekans', ascending=False).reset_index(drop=True)
        aranan_kelime = st.text_input("Arkasından ne geldiğini merak ettiğiniz kelimeyi yazın (Örn: iced, caramel):")
        
        if aranan_kelime:
            # Sadece aranan kelime ve boşluk ile başlayanları filtrele
            filtre = bigram_df['Kelime İkilisi'].str.startswith(aranan_kelime.lower() + " ")
            bigram_df = bigram_df[filtre]
        
        # Sonucu  bir tablo olarak göster
        st.dataframe(bigram_df.head(10), use_container_width=True)
    except ValueError:
        st.info("Bu yaş grubunda ikili kelime (bi-gram) oluşturacak yeterli veri bulunamadı.")

# 3. ALT KISIM - Orijinal Veri Görünümü
st.divider()
st.subheader("Süzülmüş ve Temizlenmiş Ham Veri")
st.dataframe(df_filtrelenmis.head(20), use_container_width=True)
