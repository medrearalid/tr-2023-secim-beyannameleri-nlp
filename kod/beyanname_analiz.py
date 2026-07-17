# -*- coding: utf-8 -*-
"""
Siyasi Parti Secim Beyannamesi Analiz Scripti
==============================================
Bu script, bir klasordeki siyasi parti secim beyannamelerini (PDF/TXT)
okur, Turkce dil yapisina uygun on isleme uygular ve betimsel istatistikler uretir.

Kurulum Adimlari:
-----------------
1. Gerekli kutuphaneleri yukle:
   pip install spacy pandas PyMuPDF scikit-learn

2. Turkce spaCy modelini yukle:
   python -m spacy download tr_core_news_md

   Not: Eger model uyumsuzluk hatasi verirse, su komutla kurabilirsiniz:
   pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_md/resolve/main/tr_core_news_md-1.0-py3-none-any.whl --no-deps

Kullanim:
---------
   python beyanname_analiz.py

   Script calistirildiginda, BEYANNAMELER_KLASORU degiskenindeki
   klasorden PDF/TXT dosyalarini okuyarak analiz eder.
"""

import json
import os
import re
import sys
import string
import time
from collections import Counter
from datetime import datetime

import pandas as pd
import fitz  # PyMuPDF
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from spacy.lang.tr.stop_words import STOP_WORDS as SPACY_TR_STOPWORDS

# ============================================================================
# YAPILANDIRMA
# ============================================================================

# Beyanname dosyalarinin bulundugu klasor yolu
BEYANNAMELER_KLASORU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beyannameler")

# En sik gecen kelime sayisi
TOP_N = 20

# Script surumu (agents.md changelog ile eslesir)
SCRIPT_SURUMU = "v3.3"

# Kanonik corpus dosya adi sablonu.
# Bu dosya, tum downstream analizlerin (gorsellestirme.py dahil) beslendigi
# TEK dogruluk kaynagidir. Dosya adinda surum ve uretim tarihi bulunur.
# Eski ara dosya (temizlenmis_corpus.json) artik uretilmez; eski kopyalar
# deprecated/ klasorunde saklanir.
KANONIK_CORPUS_SABLONU = "corpus_final_v1_{tarih}.json"

# Dosya-etiket eslestirme tablosu
# Her anahtar = sistemdeki etiket (parti adi)
# Her deger   = o etiket altinda birlestirilecek dosya adlarinin listesi
# Birden fazla dosya listede yer aliyorsa metinleri tek blok olarak birlestir
BEYANNAME_GRUPLARI = {
    "AKP"         : ["AKP_BEYANNAME_2023.pdf", "AKP_BEYANNAME_2023_2.pdf"],
    "CHP_IYI_Ortak": ["CHP_BEYANNAME_2023_ORTAK.pdf"],
    "DEM"         : ["DEM_BEYANNAME_2023.pdf"],
    "MHP"         : ["MHP_BEYANNAME_2023.pdf"],
}

# Turkce dolgu (stop) kelimeleri: spaCy gomulu liste + ozel eklemeler
# spaCy'nin Turkce stop-words listesini temel al
TURKCE_STOPWORDS = set(SPACY_TR_STOPWORDS)

# Siyasal metin analizinde baglam disi cok tekrar eden kelimeleri de ekle
OZEL_STOPWORDS = {
    "y\u0131l", "\u00fclke", "t\u00fcrkiye", "\u015fekil", "yeni", "b\u00fcy\u00fck",
    "parti", "milliyet", "hareket", "ak", "chp", "iyi", "dem", "mhp",
    "nin", "son", "ye", "krarli", "nda", "alt", "\u00f6n", "i\u00e7",
    "yan", "s\u0131ra", "ayn\u0131", "zaman", "v\u0131"
}
TURKCE_STOPWORDS.update(OZEL_STOPWORDS)

# POS filtreleme: sadece bu etiketlere sahip tokenlar analize dahil edilir
GECERLI_POS_ETIKETLERI = {"NOUN", "ADJ"}

# ============================================================================
# LOGLAMA
# ============================================================================

def log(seviye, mesaj):
    """Kisa ve oz terminal loglamasi."""
    zaman = datetime.now().strftime("%H:%M:%S")
    print("  [{seviye}] {zaman} | {mesaj}".format(seviye=seviye, zaman=zaman, mesaj=mesaj))
    sys.stdout.flush()


def ilerleme_goster(mevcut, toplam, etiket=""):
    """Satir ici ilerleme gosterir (ayni satiri gunceller)."""
    yuzde = int(mevcut / toplam * 100) if toplam > 0 else 0
    bar_uzunluk = 25
    dolu = int(bar_uzunluk * mevcut / toplam) if toplam > 0 else 0
    bar = "#" * dolu + "-" * (bar_uzunluk - dolu)
    sys.stdout.write("\r    [{bar}] {yuzde}% ({mevcut}/{toplam}) {etiket}   ".format(
        bar=bar, yuzde=yuzde, mevcut=mevcut, toplam=toplam, etiket=etiket))
    sys.stdout.flush()
    if mevcut >= toplam:
        sys.stdout.write("\n")
        sys.stdout.flush()


def kutuphane_surumleri():
    """Calisma aninda kullanilan kutuphane surumlerini toplar.

    Kanonik corpus meta bilgisine yazilir; boylece her ciktinin hangi
    yazilim ortaminda uretildigi belgelenmis olur (tekrarlanabilirlik).
    """
    import sklearn
    surumler = {
        "python": sys.version.split()[0],
        "spacy": spacy.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "pymupdf": getattr(fitz, "__version__", None) or str(getattr(fitz, "version", "?")),
    }
    try:
        surumler["tr_core_news_md"] = spacy.util.get_package_version("tr_core_news_md")
    except Exception:
        surumler["tr_core_news_md"] = "tespit edilemedi"
    return surumler


# ============================================================================
# 1. DOSYA OKUMA
# ============================================================================

def pdf_oku(dosya_yolu):
    """PDF dosyasini PyMuPDF (fitz) ile okur ve tum sayfalarin metnini dondurur."""
    metin_parcalari = []
    dosya_adi = os.path.basename(dosya_yolu)
    doc = fitz.open(dosya_yolu)
    toplam_sayfa = len(doc)
    for idx in range(toplam_sayfa):
        sayfa_metni = doc[idx].get_text()
        if sayfa_metni:
            metin_parcalari.append(sayfa_metni)
        sayfa_no = idx + 1
        if sayfa_no % 20 == 0 or sayfa_no == toplam_sayfa:
            ilerleme_goster(sayfa_no, toplam_sayfa, dosya_adi)
    doc.close()
    return "\n".join(metin_parcalari)


def txt_oku(dosya_yolu):
    """TXT dosyasini UTF-8 olarak okur."""
    with open(dosya_yolu, "r", encoding="utf-8") as f:
        return f.read()


def belge_oku(dosya_yolu):
    """Dosya uzantisina gore uygun okuyucuyu cagirir."""
    uzanti = os.path.splitext(dosya_yolu)[1].lower()
    if uzanti == ".pdf":
        return pdf_oku(dosya_yolu)
    elif uzanti == ".txt":
        return txt_oku(dosya_yolu)
    else:
        log("UYARI", "Desteklenmeyen dosya formati: {0}".format(uzanti))
        return ""


# ============================================================================
# 2. METIN ON ISLEME
# ============================================================================

# Manuel duzeltme sozlugu: PDF text extraction sirasinda bozulan
# spesifik kelime/kaliplarin dogru karsiliklari.
# Kural: Uzun kaliplar ONCE uygulanir (parcalarin erken eslesmesi onlenir).
# Tum karsilastirmalar kucuk harf (lower) uzerinde yapildigi icin
# sozluk anahtarlari da kucuk harfle yazilir.
METIN_DUZELTME_SOZLUGU = [
    # --- Sayfa ust/alt bilgileri ve tekrar eden baglaclar ---
    ("ortak politika mutabakat metni", " "),
    ("ortak politikalar mutabakat metni", " "),
    ("mutabakat metn", " "),
    ("vı. sektörel", " "),
    ("vı sektörel", " "),
    ("vi. sektörel", " "),
    ("vi sektörel", " "),
    ("vııı. sosyal", " "),
    ("vııı sosyal", " "),
    ("viii. sosyal", " "),
    ("viii sosyal", " "),
    ("cumhurbaşkanlığı hükümet sistemi", " "),
    ("cumhurbaşkanlığı hükümet", " "),
    ("cumhurbaşkanlık hükümet", " "),
    ("yanı sıra", " "),
    ("yan sıra", " "),
    ("aynı zamanda", " "),
    ("aynı zaman", " "),
    # --- AKP beyannamesi: dekoratif baslik bozulmalari ---
    ("yey\u00fczyilii", "y\u00fczy\u0131l\u0131"),
    ("\u00e7indo\u011fruad\u0131m", "i\u00e7in do\u011fru ad\u0131m"),
    # --- CHP/AKP: PDF hece arasi bosluk bozulmalari ---
    ("poli ti ka", "politika"),
    ("poli ti", "politika"),
    ("ti ka mutabakat", "politika mutabakat"),
    # --- Genel PDF heceleme kalintilari ---
    ("t\u00fcrki ye", "t\u00fcrkiye"),
    ("sti krarli", "istikrarl\u0131"),
    ("krarli g\u00fc\u00e7", "istikrarl\u0131 g\u00fc\u00e7"),
]


def ham_metni_onar(metin):
    """
    PDF'den cikan ham metindeki yaygin bozulmalari onarir.
    Bu fonksiyon lemmatization ONCESI, metni_temizle ONCESI calisir.

    Onarilan sorunlar:
      1. Satir sonu heceleme: 'hizmetler-\\nle' -> 'hizmetlerle'
      2. Manuel sozluk ile bilinen bozuk kalipler duzeltilir
    """
    # (1) Satir sonu heceleme birlestirme
    #     'kelime-\n devam' veya 'kelime- \n devam' -> 'kelimedevam'
    metin = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", metin)

    # (2) Manuel sozluk degisimi (buyuk/kucuk harf duyarsiz)
    #     metni_temizle() zaten lower() yapar, burada sadece bozuk
    #     kaliplari duzeltiriz. re.IGNORECASE ile eslestirme yapilir.
    for yanlis, dogru in METIN_DUZELTME_SOZLUGU:
        metin = re.sub(re.escape(yanlis), dogru, metin, flags=re.IGNORECASE)

    return metin


def metni_temizle(metin):
    """
    Ham metni temizler:
      - Kucuk harfe cevirir
      - URL adreslerini kaldirir
      - Rakamlari kaldirir
      - Noktalama isaretlerini kaldirir
      - Fazla bosluklari tek bosluga indirger
    """
    # Kucuk harf
    metin = metin.replace("I", "ı").replace("İ", "i").lower()
    # URL kaldir
    metin = re.sub(r"https?://\S+|www\.\S+", " ", metin)
    # Rakamlari kaldir
    metin = re.sub(r"\d+", " ", metin)
    # Noktalama kaldir (Turkce ozel karakterler korunur)
    ozel_noktalar = '"""\'\'\u2018\u2019\u201c\u201d\u2013\u2014...*>#'
    tum_noktalar = string.punctuation + ozel_noktalar
    metin = metin.translate(str.maketrans("", "", tum_noktalar))
    # Fazla bosluklari temizle
    metin = re.sub(r"\s+", " ", metin).strip()
    return metin


def stopwords_kaldir(tokenlar):
    """spaCy Turkce stop-words + ozel liste ile filtreleme yapar.

    Seffaflik geregi elenen token sayilari da dondurulur; boylece hicbir
    veri noktasi sessizce elenmez, her filtre adimi loglanabilir.

    Dondurur: (temiz_tokenlar, stopword_elenen, tek_karakter_elenen)
    """
    temiz = []
    stopword_elenen = 0
    tek_karakter_elenen = 0
    for t in tokenlar:
        if t in TURKCE_STOPWORDS:
            stopword_elenen += 1
        elif len(t) <= 1:
            tek_karakter_elenen += 1
        else:
            temiz.append(t)
    return temiz, stopword_elenen, tek_karakter_elenen


def lemmatize_et(nlp, metin, etiket=""):
    """spaCy kullanarak lemmatization uygular. Buyuk metinleri parcalayarak isler.

    Dondurur: (lemma_tokenlar, filtre_sayaclari)
    filtre_sayaclari, POS filtresiyle elenen token sayisini raporlar;
    boylece filtreleme adimlari denetlenebilir (gizli eleme yok).
    """
    lemma_tokenlar = []
    filtre_sayaclari = {
        "spacy_toplam_token": 0,
        "pos_filtresi_elenen": 0,
        "bos_noktalama_elenen": 0,
    }
    # 100K karakterlik parcalar halinde isle (hiz/bellek dengesi)
    parca_boyutu = 100_000
    parcalar = [metin[i:i + parca_boyutu] for i in range(0, len(metin), parca_boyutu)]
    toplam_parca = len(parcalar)
    baslangic = time.time()

    for idx, parca in enumerate(parcalar, 1):
        doc = nlp(parca)
        for token in doc:
            filtre_sayaclari["spacy_toplam_token"] += 1
            # POS filtresi: sadece NOUN ve ADJ
            if token.pos_ not in GECERLI_POS_ETIKETLERI:
                filtre_sayaclari["pos_filtresi_elenen"] += 1
                continue
            lemma = token.lemma_.strip()
            if lemma and not token.is_punct and not token.is_space:
                lemma_tokenlar.append(lemma)
            else:
                filtre_sayaclari["bos_noktalama_elenen"] += 1
        ilerleme_goster(idx, toplam_parca, etiket)

    gecen = time.time() - baslangic
    log("BILGI", "Lemmatization suresi: {0:.1f}s ({1} parca)".format(gecen, toplam_parca))
    return lemma_tokenlar, filtre_sayaclari


def on_isleme(nlp, metin, parti_adi):
    """
    Tam on isleme pipeline:
      1. Ham kelime sayisini hesapla
      2. Ham metin onarimi (heceleme + sozluk)
      3. Metin temizleme
      4. Lemmatization
      5. Stop-word kaldirma
    Dondurur: (ham_kelime_sayisi, temiz_token_listesi, filtre_ozeti)

    filtre_ozeti: her filtre adiminda elenen token sayilarinin dokumu.
    Amac: hicbir veri noktasinin sessizce elenmemesi (seffaflik ilkesi).
    """
    # Ham kelime sayisi (onarim ve temizleme ONCESI, bosluk ayrimiyla)
    ham_kelimeler = metin.split()
    ham_kelime_sayisi = len(ham_kelimeler)

    log("BILGI", "Ham metin onariliyor (heceleme + sozluk): {0}".format(parti_adi))
    metin = ham_metni_onar(metin)

    log("BILGI", "Metin temizleniyor: {0}".format(parti_adi))
    temiz_metin = metni_temizle(metin)

    log("BILGI", "Lemmatization uygulan\u0131yor: {0} ({1} karakter)".format(
        parti_adi, len(temiz_metin)))
    lemma_tokenlar, filtre_sayaclari = lemmatize_et(nlp, temiz_metin, etiket=parti_adi)

    log("BILGI", "Stop-words kald\u0131r\u0131l\u0131yor: {0}".format(parti_adi))
    temiz_tokenlar, stopword_elenen, tek_karakter_elenen = stopwords_kaldir(lemma_tokenlar)

    filtre_ozeti = {
        "ham_kelime": ham_kelime_sayisi,
        "spacy_toplam_token": filtre_sayaclari["spacy_toplam_token"],
        "pos_filtresi_elenen": filtre_sayaclari["pos_filtresi_elenen"],
        "bos_noktalama_elenen": filtre_sayaclari["bos_noktalama_elenen"],
        "stopword_elenen": stopword_elenen,
        "tek_karakter_elenen": tek_karakter_elenen,
        "islenmis_token": len(temiz_tokenlar),
    }
    log("BILGI", ("Filtre dokumu [{0}]: spaCy token={1} | POS elenen={2} | "
                  "stopword elenen={3} | tek karakter elenen={4} | kalan={5}").format(
        parti_adi,
        filtre_ozeti["spacy_toplam_token"],
        filtre_ozeti["pos_filtresi_elenen"],
        filtre_ozeti["stopword_elenen"],
        filtre_ozeti["tek_karakter_elenen"],
        filtre_ozeti["islenmis_token"]))

    return ham_kelime_sayisi, temiz_tokenlar, filtre_ozeti


# ============================================================================
# 3. ANALIZ VE ISTATISTIKLER
# ============================================================================

def betimsel_istatistik_tablosu(sonuclar):
    """Parti bazli betimsel istatistik DataFrame olusturur."""
    df = pd.DataFrame(sonuclar)
    # Toplam satiri ekle
    ham_toplam = df["Ham Kelime Sayisi"].sum()
    islem_toplam = df["Islenmis Token Sayisi"].sum()
    dosya_toplam = df["Kaynak Dosya Sayisi"].sum()
    azalma = round((1 - islem_toplam / ham_toplam) * 100, 2) if ham_toplam > 0 else 0
    toplam = pd.DataFrame([{
        "Parti / Belge": "--- TOPLAM ---",
        "Kaynak Dosya Sayisi": dosya_toplam,
        "Ham Kelime Sayisi": ham_toplam,
        "Islenmis Token Sayisi": islem_toplam,
        "Azalma Orani (%)": azalma,
    }])
    df = pd.concat([df, toplam], ignore_index=True)
    return df


def en_sik_kelimeler(tum_tokenlar, n=TOP_N):
    """Tum derlemi kapsayan en sik N kelimeyi hesaplar."""
    sayac = Counter(tum_tokenlar)
    en_sik = sayac.most_common(n)
    df = pd.DataFrame(en_sik, columns=["Kelime", "Frekans"])
    df.index = range(1, len(df) + 1)
    df.index.name = "Sira"
    return df


def tfidf_analizi(parti_token_sozlugu, top_n=15):
    """
    Parti bazli TF-IDF analizi.
    Her parti icin o partiyi diger partilerden en cok ayiran
    ilk top_n kelimeyi ve TF-IDF skorlarini hesaplar.

    Parametreler:
        parti_token_sozlugu: {"AKP": ["kelime1", "kelime2", ...], ...}
        top_n: Her parti icin dondurulecek kelime sayisi

    Dondurur: {parti_adi: pd.DataFrame(Kelime, TF-IDF Skoru)}
    """
    etiketler = list(parti_token_sozlugu.keys())
    # Her partinin token listesini tek bir string belge olarak birlestir
    belgeler = [" ".join(parti_token_sozlugu[e]) for e in etiketler]

    # TfidfVectorizer: her parti bir belge.
    # Parametreler tekrarlanabilirlik icin ACIKCA yazilmistir (bunlar
    # scikit-learn varsayilanlaridir, davranis degismez):
    #   norm="l2"         -> belge vektorleri L2 normuna gore normalize edilir
    #   use_idf=True, smooth_idf=True -> yumusatilmis ters belge sikligi
    #                        (sifira bolunmeyi onler; tez Bolum 2.4 ile uyumlu)
    #   sublinear_tf=False -> ham terim frekansi (log olcekleme yok)
    #   max_features=None  -> kelime dagarcigi SINIRLANMAZ. Derlem yalnizca
    #                        4 belgeden olustugu icin sinir gereksizdir;
    #                        keyfi bir kesme (cutoff) secim yanliligi yaratirdi.
    vectorizer = TfidfVectorizer(norm="l2", use_idf=True, smooth_idf=True,
                                 sublinear_tf=False, max_features=None)
    tfidf_matris = vectorizer.fit_transform(belgeler)
    kelimeler = vectorizer.get_feature_names_out()

    sonuc = {}
    for idx, etiket in enumerate(etiketler):
        # Bu partiye ait TF-IDF skorlarini al
        skorlar = tfidf_matris[idx].toarray().flatten()
        # Skorlara gore sirala (buyukten kucuge)
        skor_sirali = sorted(zip(kelimeler, skorlar), key=lambda x: x[1], reverse=True)
        # Ilk top_n kelimeyi al
        ust_n = skor_sirali[:top_n]
        df = pd.DataFrame(ust_n, columns=["Kelime", "TF-IDF Skoru"])
        df["TF-IDF Skoru"] = df["TF-IDF Skoru"].round(4)
        df.index = range(1, len(df) + 1)
        df.index.name = "Sira"
        sonuc[etiket] = df

    return sonuc


def bigram_analizi(parti_token_sozlugu, top_n=10):
    """
    Parti bazli bigram (ikili kelime obegi) analizi.
    Her parti icin en sik kullanilan ilk top_n bigrami hesaplar.

    Ham frekansa ek olarak NORMALIZE frekans da hesaplanir:
        normalize = (ham_frekans / parti_toplam_islenmis_token) * 10000
    Payda olarak partinin TOPLAM islenmis token sayisi kullanilir
    (ilk-N bigram alt kumesinin toplami DEGIL). Gerekce:
      1. Partiler arasi metin uzunlugu farki (AKP ~73K vs DEM ~7K token)
         ham sayimlari dogrudan karsilastirilamaz kilar; 10.000 token
         basina oran bu sapmayi giderir (radar/cubuk grafikle ayni yontem).
      2. Ilk-N alt kumesi uzerinden grup toplami almak, paydayi partiden
         partiye farkli ve keyfi bicimde kuculterek yanlilik yaratirdi.

    Parametreler:
        parti_token_sozlugu: {"AKP": ["kelime1", "kelime2", ...], ...}
        top_n: Her parti icin dondurulecek bigram sayisi

    Dondurur: {parti_adi: pd.DataFrame(Bigram, Frekans, Toplam Token,
                                       Frekans (10k Token Basina))}
    """
    sonuc = {}
    for etiket, tokenlar in parti_token_sozlugu.items():
        toplam_token = len(tokenlar)
        belge = " ".join(tokenlar)
        vectorizer = CountVectorizer(ngram_range=(2, 2))
        bigram_matris = vectorizer.fit_transform([belge])
        bigramlar = vectorizer.get_feature_names_out()
        frekanslar = bigram_matris.toarray().flatten()

        # Frekanslara gore sirala
        skor_sirali = sorted(zip(bigramlar, frekanslar), key=lambda x: x[1], reverse=True)
        ust_n = skor_sirali[:top_n]
        df = pd.DataFrame(ust_n, columns=["Bigram", "Frekans"])
        df["Toplam Token"] = toplam_token
        if toplam_token > 0:
            df["Frekans (10k Token Basina)"] = (
                df["Frekans"] / toplam_token * 10000).round(2)
        else:
            df["Frekans (10k Token Basina)"] = 0.0
        df.index = range(1, len(df) + 1)
        df.index.name = "Sira"
        sonuc[etiket] = df

    return sonuc


# ============================================================================
# 4. ANA IS AKISI
# ============================================================================

def main():
    """Ana fonksiyon: tum pipeline calistirir."""
    print()
    print("=" * 60)
    print("  SECIM BEYANNAMESI METIN MADENCILIGI ANALIZI")
    print("=" * 60)
    print()

    # Klasor kontrolu
    if not os.path.isdir(BEYANNAMELER_KLASORU):
        log("HATA", "Klasor bulunamadi: {0}".format(BEYANNAMELER_KLASORU))
        log("BILGI", "Lutfen 'beyannameler' adinda bir klasor olusturup")
        log("BILGI", "PDF veya TXT dosyalarini bu klasore yerlestirin.")
        return

    log("BILGI", "{0} parti/grup tanimli.".format(len(BEYANNAME_GRUPLARI)))
    print()

    # spaCy modelini yukle (sadece lemmatizer + tok2vec aktif, geri kalan KAPALI)
    log("BILGI", "spaCy Turkce modeli yukleniyor (tr_core_news_md)...")
    try:
        nlp = spacy.load("tr_core_news_md",
                         disable=["parser", "ner", "attribute_ruler"])
    except OSError:
        log("HATA", "spaCy modeli bulunamadi!")
        log("BILGI", "Su komutla yukleyebilirsiniz:")
        log("BILGI", "  python -m spacy download tr_core_news_md")
        return
    # Buyuk metinler icin karakter sinirini kaldir
    nlp.max_length = 5_000_000
    log("BASARILI", "Model yuklendi (aktif: {0})".format(", ".join(nlp.pipe_names)))
    print()

    # Sonuclari toplayacak yapilar
    sonuclar = []
    tum_tokenlar = []
    parti_token_sozlugu = {}  # Parti bazli token listesi (TF-IDF ve Bigram icin)
    filtre_ozetleri = {}      # Parti bazli filtre dokumu (seffaflik kaydi)

    # Her grubu (parti etiketini) isle
    gruplar = list(BEYANNAME_GRUPLARI.items())
    for i, (etiket, dosya_listesi) in enumerate(gruplar, 1):

        print("  -- Grup {0}/{1}: {2} --".format(i, len(gruplar), etiket))

        # Gruptaki tum dosyalari oku ve tek blokta birlestir
        birlesmis_metin_parcalari = []
        dosya_sayisi = len(dosya_listesi)

        for dosya_adi in dosya_listesi:
            dosya_yolu = os.path.join(BEYANNAMELER_KLASORU, dosya_adi)

            if not os.path.isfile(dosya_yolu):
                log("UYARI", "Dosya bulunamadi, atlaniyor: {0}".format(dosya_adi))
                continue

            log("BILGI", "{0} okunuyor...".format(dosya_adi))
            parca = belge_oku(dosya_yolu)

            if parca:
                birlesmis_metin_parcalari.append(parca)
            else:
                log("UYARI", "{0} bos veya okunamadi, atlaniyor.".format(dosya_adi))

        if not birlesmis_metin_parcalari:
            log("UYARI", "[{0}] icin okunabilir dosya yok, grup atlaniyor.".format(etiket))
            print()
            continue

        if dosya_sayisi > 1:
            log("BILGI", "{0} dosya tek metin blogunda birlestiriliyor -> [{1}]".format(
                len(birlesmis_metin_parcalari), etiket))

        # Birlestirilmis ham metin
        ham_metin = "\n\n".join(birlesmis_metin_parcalari)

        # On isleme
        ham_kelime_sayisi, temiz_tokenlar, filtre_ozeti = on_isleme(nlp, ham_metin, etiket)
        filtre_ozetleri[etiket] = filtre_ozeti

        # Sonuclari kaydet
        azalma_orani = round(
            (1 - len(temiz_tokenlar) / ham_kelime_sayisi) * 100, 2
        ) if ham_kelime_sayisi > 0 else 0

        sonuclar.append({
            "Parti / Belge": etiket,
            "Kaynak Dosya Sayisi": len(birlesmis_metin_parcalari),
            "Ham Kelime Sayisi": ham_kelime_sayisi,
            "Islenmis Token Sayisi": len(temiz_tokenlar),
            "Azalma Orani (%)": azalma_orani,
        })

        tum_tokenlar.extend(temiz_tokenlar)
        parti_token_sozlugu[etiket] = temiz_tokenlar
        log("BASARILI", "[{0}] tamamlandi: {1} ham -> {2} token".format(
            etiket, ham_kelime_sayisi, len(temiz_tokenlar)))
        print()

    if not sonuclar:
        log("HATA", "Hicbir belge islenemedi.")
        return

    # -- SONUCLAR --
    print("=" * 60)
    print("  BETIMSEL ISTATISTIKLER (Tablo 3.1)")
    print("=" * 60)
    print()

    df_istatistik = betimsel_istatistik_tablosu(sonuclar)
    print(df_istatistik.to_string(index=False))
    print()

    print("=" * 60)
    print("  EN SIK GECEN ILK {0} KELIME (Tum Derlem)".format(TOP_N))
    print("=" * 60)
    print()

    df_frekans = en_sik_kelimeler(tum_tokenlar, TOP_N)
    print(df_frekans.to_string())
    print()

    # -- FAZ 2: TF-IDF ANALIZI --
    print("=" * 60)
    print("  TF-IDF ANALIZI (Parti Bazli Ilk 15 Karakteristik Kelime)")
    print("=" * 60)
    print()

    log("BILGI", "TF-IDF hesaplaniyor (TfidfVectorizer)...")
    tfidf_sonuclari = tfidf_analizi(parti_token_sozlugu, top_n=15)
    log("BASARILI", "TF-IDF analizi tamamlandi.")
    print()

    for etiket, df_tfidf in tfidf_sonuclari.items():
        print("  --- {0} ---".format(etiket))
        print(df_tfidf.to_string())
        print()

    # -- FAZ 2: BIGRAM ANALIZI --
    print("=" * 60)
    print("  BIGRAM ANALIZI (Parti Bazli Ilk 10 Ikili Kelime Obegi)")
    print("=" * 60)
    print()

    log("BILGI", "Bigram frekanslari hesaplaniyor (CountVectorizer)...")
    bigram_sonuclari = bigram_analizi(parti_token_sozlugu, top_n=10)
    log("BASARILI", "Bigram analizi tamamlandi.")
    print()

    for etiket, df_bigram in bigram_sonuclari.items():
        print("  --- {0} ---".format(etiket))
        print(df_bigram.to_string())
        print()

    # -- SONUCLARI CSV OLARAK KAYDET --
    cikti_klasoru = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sonuclar")
    os.makedirs(cikti_klasoru, exist_ok=True)

    istatistik_yolu = os.path.join(cikti_klasoru, "betimsel_istatistikler.csv")
    frekans_yolu = os.path.join(cikti_klasoru, "en_sik_kelimeler.csv")
    tfidf_yolu = os.path.join(cikti_klasoru, "tfidf_skorlari.csv")
    bigram_yolu = os.path.join(cikti_klasoru, "bigram_frekanslari.csv")

    df_istatistik.to_csv(istatistik_yolu, index=False, encoding="utf-8-sig")
    df_frekans.to_csv(frekans_yolu, encoding="utf-8-sig")

    # TF-IDF sonuclarini tek CSV'de birlestir (Parti sutunu ile)
    tfidf_birlesmis = []
    for etiket, df_t in tfidf_sonuclari.items():
        gecici = df_t.copy()
        gecici.insert(0, "Parti", etiket)
        tfidf_birlesmis.append(gecici)
    pd.concat(tfidf_birlesmis, ignore_index=True).to_csv(
        tfidf_yolu, index=False, encoding="utf-8-sig")

    # Bigram sonuclarini tek CSV'de birlestir (Parti sutunu ile)
    bigram_birlesmis = []
    for etiket, df_b in bigram_sonuclari.items():
        gecici = df_b.copy()
        gecici.insert(0, "Parti", etiket)
        bigram_birlesmis.append(gecici)
    pd.concat(bigram_birlesmis, ignore_index=True).to_csv(
        bigram_yolu, index=False, encoding="utf-8-sig")

    # Filtre dokumunu ayri CSV olarak kaydet (seffaflik kaydi):
    # her partide hangi filtre adiminin kac token eledigi gorulur.
    filtre_yolu = os.path.join(cikti_klasoru, "filtre_dokumu.csv")
    df_filtre = pd.DataFrame(filtre_ozetleri).T
    df_filtre.index.name = "Parti / Belge"
    df_filtre.to_csv(filtre_yolu, encoding="utf-8-sig")

    log("BASARILI", "Tum tablolar CSV olarak kaydedildi: {0}".format(cikti_klasoru))
    print()

    # -- KANONIK CORPUS DOSYASI (TEK DOGRULUK KAYNAGI) --
    # Gorsellestirme modulu dahil TUM downstream analizler yalnizca bu
    # dosyadan beslenir. Dosya iki bolumden olusur:
    #   "_meta"  : uretim zamani, script surumu, kutuphane surumleri,
    #              filtre dokumu (denetlenebilirlik icin)
    #   "corpus" : {parti_etiketi: temizlenmis token stringi}
    # Eski ad (temizlenmis_corpus.json) artik uretilmez.
    corpus_json = {}
    for etiket, tokenlar in parti_token_sozlugu.items():
        corpus_json[etiket] = " ".join(tokenlar)

    kanonik_ad = KANONIK_CORPUS_SABLONU.format(
        tarih=datetime.now().strftime("%Y-%m-%d"))
    json_yolu = os.path.join(cikti_klasoru, kanonik_ad)

    kanonik_icerik = {
        "_meta": {
            "olusturma_zamani": datetime.now().isoformat(timespec="seconds"),
            "script_surumu": SCRIPT_SURUMU,
            "kutuphane_surumleri": kutuphane_surumleri(),
            "spacy_pipeline": list(nlp.pipe_names),
            "spacy_devre_disi": ["parser", "ner", "attribute_ruler"],
            "pos_filtresi": sorted(GECERLI_POS_ETIKETLERI),
            "spacy_stopword_sayisi": len(SPACY_TR_STOPWORDS),
            "ozel_stopword_listesi": sorted(OZEL_STOPWORDS),
            "filtre_dokumu": filtre_ozetleri,
        },
        "corpus": corpus_json,
    }

    with open(json_yolu, "w", encoding="utf-8") as f:
        json.dump(kanonik_icerik, f, ensure_ascii=False, indent=2)

    log("BASARILI", "Kanonik corpus kaydedildi: {0}".format(json_yolu))
    print()
    print("=" * 60)
    print("  ANALIZ TAMAMLANDI")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
