# -*- coding: utf-8 -*-
"""
Siyasi Parti Secim Beyannamesi - Gorsellestirme Modulu
======================================================
Bu script, beyanname_analiz.py tarafindan uretilen kanonik corpus dosyasini
(corpus_final_v*.json, en guncel tarih damgali surum otomatik secilir) okuyarak
gorsellestirme analizleri uretir.

Bagimliliklari: pandas, scikit-learn, matplotlib, seaborn, numpy, plotly, kaleido
Not: teknoloji_duygu_analizi() fonksiyonu orijinal PDF'leri okudugu icin
     ek olarak PyMuPDF (fitz) gerektirir. Sankey diyagrami plotly + kaleido ister.

Kullanim:
---------
   python gorsellestirme.py

Ciktilar (tumu sonuclar/ klasorune, 300 DPI):
---------
   pca_ideolojik_uzay.png            - PCA ideolojik uzay haritasi (scatter)
   radar_kavramlar.png               - Kavram radar (orumcek) grafigi
   bar_kavramlar.png                 - Kavram gruplu cubuk grafigi
   lexicon_eksen_haritasi.png        - Inovasyon vs. Regulasyon eksen haritasi
   duygu_analizi_teknoloji_modern.png- Teknoloji odakli duygu analizi (yigilmis cubuk)
   duygu_sankey_teknoloji.png        - Teknoloji odakli soylem akis haritasi (Sankey)
"""

import json
import os
import re
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # GUI olmadan calistirmak icin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# ============================================================================
# YAPILANDIRMA
# ============================================================================

# Dosya yollari
PROJE_KOKU = os.path.dirname(os.path.abspath(__file__))
SONUCLAR_KLASORU = os.path.join(PROJE_KOKU, "sonuclar")

# Kanonik corpus dosya deseni.
# beyanname_analiz.py, surum + tarih damgali TEK dogruluk kaynagi uretir
# (orn. corpus_final_v1_2026-07-02.json). Tum analizler YALNIZCA bu
# dosyadan beslenir; eski ad (temizlenmis_corpus.json) kullanilmaz.
KANONIK_CORPUS_DESENI = "corpus_final_v*.json"

# ----------------------------------------------------------------------------
# BIRLESIK AKADEMIK RENK PALETI VE GORSEL STILI
# ----------------------------------------------------------------------------
# Tum grafikler tek, renk korlugune duyarli ve baski dostu bir paletten beslenir;
# boylece tezdeki tum sekiller gorsel olarak tutarlidir.
# Palet: Okabe & Ito (2008), "Color Universal Design" (CUD) onerisine dayanir.
# ----------------------------------------------------------------------------

# Kategorik palet (parti renkleri). Anahtarlar kanonik corpus JSON'undaki
# "corpus" sozlugunun etiketleriyle birebir aynidir.
PARTI_RENKLERI = {
    "AKP":          "#E69F00",  # Turuncu
    "CHP_IYI_Ortak":"#D55E00",  # Kiremit (vermillion)
    "DEM":          "#CC79A7",  # Mor-pembe
    "MHP":          "#0072B2",  # Mavi
}

# Duygu kategorileri icin tutarli yesil-gri-kiremit alt-palet.
DUYGU_RENKLERI = {
    "Pozitif": "#009E73",  # Yesil (bluish green)
    "Nötr":    "#BFBFBF",  # Notr gri
    "Negatif": "#D55E00",  # Kiremit (vermillion)
}

# Yardimci renkler
RENK_METIN = "#222222"
RENK_IZGARA = "#CCCCCC"


def akademik_stil_uygula():
    """Tum matplotlib figurleri icin tutarli, yayima uygun stil ayarlar.

    Tek noktadan font, izgara, kenarlik ve arka plan ayarini yapar; boylece
    her grafikte ayri ayri stil tanimlamaya gerek kalmaz ve gorsel tutarlilik
    saglanir. main() basinda bir kez cagrilir.
    """
    sns.set_theme(style="white", context="notebook")
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "savefig.facecolor":  "white",
        "font.family":        "DejaVu Sans",
        "font.size":          11,
        "axes.titlesize":     14,
        "axes.titleweight":   "bold",
        "axes.labelsize":     12,
        "axes.labelcolor":    RENK_METIN,
        "text.color":         RENK_METIN,
        "axes.edgecolor":     "#444444",
        "axes.linewidth":     0.8,
        "axes.grid":          False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "xtick.color":        RENK_METIN,
        "ytick.color":        RENK_METIN,
        "legend.fontsize":    10,
        "legend.frameon":     False,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })

# Radar grafik icin aranacak kavramlar (gosterilecek nihai etiketler)
RADAR_KAVRAMLARI = [
    "teknoloji", "dijital", "yapay zek\u00e2", "g\u00fcvenlik",
    "milli", "sosyal", "liyakat", "adalet"
]

# --- Tematik kelime ailesi / composite skor yapilandirmasi (Bolum 3.x) ---
# Tekil "teknoloji" kelimesi yerine iliskili kavram ailesinin toplamina
# dayali bilesik (composite) skor hesaplanir.
#
# "yapay zeka" hakkindaki tokenizasyon karari: lemmatizasyon sonrasi bu
# kavram corpus icinde IKI AYRI lemma ("yapay" + "zeka"/"zek\u00e2") olarak yer
# alir; tek-kelime ("yapayzeka") bicimi corpus icinde HIC gecmez. Bu nedenle
# kavram, radar/cubuk grafiklerdeki mevcut yaklasimla TUTARLI olarak
# ngram_range=(1,2) uzerinden iki kelimelik BIGRAM bicimiyle sayilir ve
# iki imla varyanti ("yapay zeka" / "yapay zek\u00e2") tek kavramda toplanir.
# "yapay" ve "zeka" unigramlari tema listesinde OLMADIGI icin cifte sayim
# olusmaz.
COMPOSITE_TEMA_UNIGRAMLAR = ["teknoloji", "dijital", "veri", "algoritma", "siber"]
COMPOSITE_TEMA_BIGRAMLAR = ["yapay zeka", "yapay zek\u00e2"]

# --- Dusuk orneklem esigi (istatistiksel guvenilirlik uyarisi) ---
# n < MIN_GUVENILIR_N olan kirilimlarda (ozellikle DEM) yuzde oranlari
# istatistiksel olarak guvenilir kabul edilmez. Uyari otomatik olarak
# (1) konsola, (2) uretilen grafik dipnotlarina, (3) sonuc CSV dosyasina
# yansitilir.
MIN_GUVENILIR_N = 10

# --- Duygu sozlugu elle kodlama dogrulamasi ---
# random_state=42: tekrarlanabilirlik icin sabitlendi; ayni corpus ile her
# calistirmada ayni 60 cumlelik dogrulama orneklemi uretilir.
DOGRULAMA_ORNEKLEM_N = 60
DOGRULAMA_RANDOM_STATE = 42

# Lexicon tabanli eksen analizi icin sozlukler (Bolum 3.3.1)
INOVASYON_SOZLUGU = [
    "inovasyon", "b\u00fcy\u00fcme", "giri\u015fimci", "giri\u015fimcilik",
    "rekabet", "yat\u0131r\u0131m", "\u00fcretim", "kalk\u0131nma"
]
# "h\u00e2k": spaCy lemmatizer "hak/hakki/haklari" cekimlerini tutarsizca bu
# diyakritikli lemmaya indirgiyor (bkz. "temel h\u00e2k \u00f6zg\u00fcrl\u00fck" baglaminda -
# gercek anlami "hak"). CountVectorizer(vocabulary=...) strip_accents=None
# oldugundan "hak" ile "h\u00e2k" ayri sayilir; tum partilerde ~2-4x sakl\u0131 kayip.
REGULASYON_SOZLUGU = [
    "reg\u00fclasyon", "etik", "\u015feffafl\u0131k", "emek", "hak", "h\u00e2k",
    "ayr\u0131mc\u0131l\u0131k", "denetim", "d\u00fczenleme", "e\u015fitlik"
]

# --- Teknoloji odakli duygu analizi yapilandirmasi (Bolum 3.3.2) ---
# Bu fonksiyon orijinal PDF'leri yeniden okur; etiket-dosya eslestirmesi
# beyanname_analiz.py icindeki BEYANNAME_GRUPLARI ile birebir aynidir.
DUYGU_BEYANNAME_GRUPLARI = {
    "AKP"          : ["AKP_BEYANNAME_2023.pdf", "AKP_BEYANNAME_2023_2.pdf"],
    "CHP_IYI_Ortak": ["CHP_BEYANNAME_2023_ORTAK.pdf"],
    "DEM"          : ["DEM_BEYANNAME_2023.pdf"],
    "MHP"          : ["MHP_BEYANNAME_2023.pdf"],
}

# Duygu analizinde hedeflenen teknoloji kavramlari (bu kelimeleri iceren
# cumleler analize alinir).
DUYGU_HEDEF_KELIMELER = ["teknoloji", "dijital", "yapay zeka", "yapay zek\u00e2"]

# Lexicon tabanli duygu sozlukleri.
DUYGU_POZITIF_SOZLUK = [
    "kalk\u0131nma", "f\u0131rsat", "yenilik", "geli\u015fim", "b\u00fcy\u00fcme", "lider",
    "ba\u015far\u0131", "destek", "yat\u0131r\u0131m", "vizyon", "at\u0131l\u0131m", "potansiyel",
    "g\u00fc\u00e7", "\u00fcretim", "katk\u0131", "kolayl\u0131k", "y\u00fckseli\u015f", "hedef",
    "\u00e7\u00f6z\u00fcm", "\u00f6nc\u00fc", "yerli", "milli", "kapsaml\u0131", "g\u00fcvenli",
    "ileri", "yeni", "\u00e7a\u011f", "devrim", "ilerleme", "m\u00fckemmel", "avantaj", "fayda"
]
DUYGU_NEGATIF_SOZLUK = [
    "tehdit", "risk", "tehlike", "sorun", "kriz", "gerileme", "zarar",
    "ba\u011f\u0131ml\u0131l\u0131k", "ihlal", "sald\u0131r\u0131", "h\u0131rs\u0131zl\u0131k", "su\u00e7",
    "korku", "a\u00e7\u0131k", "yasad\u0131\u015f\u0131", "s\u0131z\u0131nt\u0131", "engel", "k\u0131s\u0131tlama",
    "endi\u015fe", "zay\u0131f", "yetersiz", "k\u00f6t\u00fc", "istismar", "ihl\u00e2l", "k\u0131r\u0131lgan",
    "savunmas\u0131z"
]

# ============================================================================
# LOGLAMA
# ============================================================================

def log(seviye, mesaj):
    """Kisa ve oz terminal loglamasi."""
    zaman = datetime.now().strftime("%H:%M:%S")
    print("  [{seviye}] {zaman} | {mesaj}".format(seviye=seviye, zaman=zaman, mesaj=mesaj))
    sys.stdout.flush()


# ============================================================================
# VERI YUKLEME
# ============================================================================

def corpus_yukle(sonuclar_klasoru):
    """Kanonik corpus dosyasini (corpus_final_v*.json) bulur ve okur.

    Dosya beyanname_analiz.py tarafindan uretilir; "_meta" bolumunde
    uretim zamani, script surumu ve kutuphane surumleri bulunur.
    Birden fazla surum varsa en son degistirilen dosya secilir ve
    hangi dosyanin kullanildigi loglanir (denetlenebilirlik).

    Dondurur: (corpus_dict, meta_dict) veya hata durumunda (None, None)
    """
    import glob
    adaylar = sorted(
        glob.glob(os.path.join(sonuclar_klasoru, KANONIK_CORPUS_DESENI)),
        key=os.path.getmtime)
    if not adaylar:
        log("HATA", "Kanonik corpus bulunamadi: {0}".format(
            os.path.join(sonuclar_klasoru, KANONIK_CORPUS_DESENI)))
        log("BILGI", "Lutfen once beyanname_analiz.py scriptini calistirin.")
        return None, None

    secilen = adaylar[-1]
    with open(secilen, "r", encoding="utf-8") as f:
        icerik = json.load(f)

    meta = icerik.get("_meta", {})
    corpus = icerik.get("corpus")
    if not corpus:
        log("HATA", "Corpus dosyasinda 'corpus' bolumu yok: {0}".format(secilen))
        return None, None

    log("BASARILI", "{0} parti corpusu yuklendi: {1}".format(
        len(corpus), os.path.basename(secilen)))
    log("BILGI", "Corpus uretimi: {0} | script {1}".format(
        meta.get("olusturma_zamani", "?"), meta.get("script_surumu", "?")))
    return corpus, meta


# ============================================================================
# A) IDEOLOJIK UZAY HARITASI (PCA)
# ============================================================================

def ideolojik_uzay_haritasi(corpus, cikti_yolu):
    """
    TF-IDF matrisi uzerinde PCA (2 boyut) uygulayarak
    partilerin ideolojik uzaydaki konumlarini scatter plot olarak cizer.
    """
    etiketler = list(corpus.keys())
    belgeler = [corpus[e] for e in etiketler]

    # TF-IDF vektorlestirme. Parametreler beyanname_analiz.py icindeki
    # tfidf_analizi ile birebir ayni ve tekrarlanabilirlik icin aciktir
    # (scikit-learn varsayilanlari; max_features=None -> sinir yok, cunku
    # derlem 4 belgedir ve kesme secim yanliligi yaratirdi).
    vectorizer = TfidfVectorizer(norm="l2", use_idf=True, smooth_idf=True,
                                 sublinear_tf=False, max_features=None)
    tfidf_matris = vectorizer.fit_transform(belgeler)

    # PCA ile 2 boyuta indirgeme.
    # svd_solver="full": bu matris boyutunda (4 x ~20K) varsayilan "auto"
    # secimi randomized SVD calistirir ve seed verilmediginden calistirmalar
    # arasinda isaret/koordinat farki dogabilirdi. "full" kesin ve
    # deterministik cozum verir; matris kucuk oldugundan maliyeti yoktur.
    #
    # StandardScaler BILINCLI olarak uygulanmamistir: TF-IDF vektorleri
    # zaten L2 normuna gore normalize edilmistir ve yalnizca 4 gozlem
    # varken her terimi birim varyansa olceklemek, tek belgede gecen nadir
    # terimlerin etkisini yapay olarak sisirirdi. TF-IDF matrisine dogrudan
    # PCA uygulamak metin madenciliginde yerlesik pratiktir (LSA benzeri).
    pca = PCA(n_components=2, svd_solver="full")
    koordinatlar = pca.fit_transform(tfidf_matris.toarray())
    # PC1 ve PC2 bilesen yukleri (loadings): hangi kelimelerin her eksene
    # ne yonde ve ne kadar katki verdigini gosterir.
    kelimeler = vectorizer.get_feature_names_out()

    print()
    print("  " + "=" * 50)
    print("  PCA BILESEN YUKLERI (Loadings)")
    print("  " + "=" * 50)
    for comp_idx, comp_adi in enumerate(["PC1", "PC2"]):
        yukler = pca.components_[comp_idx]
        sirali = sorted(zip(kelimeler, yukler), key=lambda x: x[1])
        print(f"\n  {comp_adi} - NEGATIF UC (ilk 15):")
        for kelime, deger in sirali[:15]:
            print(f"    {kelime}: {deger:.4f}")
        print(f"\n  {comp_adi} - POZITIF UC (ilk 15):")
        for kelime, deger in sirali[-15:][::-1]:
            print(f"    {kelime}: {deger:.4f}")
    print()
    # Konsola koordinatlari yazdir
    print()
    print("  " + "=" * 50)
    print("  PCA KOORDINATLARI (Ideolojik Uzay)")
    print("  " + "=" * 50)
    df_koord = pd.DataFrame(koordinatlar, columns=["PC1 (X)", "PC2 (Y)"], index=etiketler)
    df_koord.index.name = "Parti"
    print(df_koord.to_string())
    print()
    print("  Aciklanan Varyans: PC1={0:.2%}, PC2={1:.2%}".format(
        pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]))
    print()

    # --- Grafik ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # Arka plan ve cerceve stilleri
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for idx, etiket in enumerate(etiketler):
        x, y = koordinatlar[idx]
        renk = PARTI_RENKLERI.get(etiket, "#333333")

        # Nokta
        ax.scatter(x, y, color=renk, s=220, zorder=5, edgecolors="white", linewidths=1.5)

        # Parti ismi
        ax.annotate(
            etiket,
            (x, y),
            textcoords="offset points",
            xytext=(12, 8),
            fontsize=12,
            fontweight="bold",
            color=renk,
            ha="left",
        )

    # Baslik ve eksen etiketleri
    ax.set_title(
        "Ideolojik Uzay Haritasi (PCA - TF-IDF)",
        fontsize=16, fontweight="bold", pad=20
    )
    ax.set_xlabel(
        "PC1 ({0:.1%} varyans)".format(pca.explained_variance_ratio_[0]),
        fontsize=12, labelpad=10
    )
    ax.set_ylabel(
        "PC2 ({0:.1%} varyans)".format(pca.explained_variance_ratio_[1]),
        fontsize=12, labelpad=10
    )

    # Grid hafifletme
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    log("BASARILI", "PCA grafigi kaydedildi: {0}".format(cikti_yolu))


# ============================================================================
# B) KAVRAM RADAR GRAFIGI (ORUMCEK GRAFIGI)
# ============================================================================

def kavram_radar_grafigi(corpus, kavramlar, cikti_yolu):
    """
    Belirli kavramlarin parti bazli oransal frekanslarini (her 10.000
    kelimede kullanim sikligi) hesaplayarak radar (orumcek) grafik cizer.

    Ham sayimlar yerine oransal frekans kullanilir cunku parti
    beyannamelerinin metin uzunluklari birbirinden cok farklidir
    (orn. AKP ~73K token, DEM ~7K token). Ham sayim bu durumda
    'Belge Uzunlugu Sapmasi' yaratir.
    """
    etiketler = list(corpus.keys())
    belgeler = [corpus[e] for e in etiketler]

    # Her partinin toplam token sayisi
    token_sayilari = [len(corpus[e].split()) for e in etiketler]

    # Vektorel sayim icin kullanilacak genisletilmis sozluk (unigram + bigram)
    arama_kavramlari = [
        "teknoloji", "dijital", "yapay zeka", "yapay zek\u00e2",
        "g\u00fcvenlik", "milli", "mill\u00ee", "sosyal", "liyakat", "adalet", "ad\u00e2let"
    ]

    # Sadece istenen kavramlari sayacak CountVectorizer (ngram_range=(1,2) olmali ki bigramlari da bulsun)
    vectorizer = CountVectorizer(vocabulary=arama_kavramlari, ngram_range=(1, 2))
    count_matris = vectorizer.fit_transform(belgeler)
    ham_frekans_tum = count_matris.toarray()  # (parti_sayisi x arama_kavram_sayisi)

    df_ham = pd.DataFrame(ham_frekans_tum, columns=arama_kavramlari)
    
    # 'yapay zeka' ve 'yapay zekâ' frekanslarini toplayarak tek bir 'yapay zekâ' sutunu olustur
    df_ham["yapay zek\u00e2"] = df_ham["yapay zeka"] + df_ham["yapay zek\u00e2"]

    # Osmanli/duzeltme isaretli imla varyantlarini (\u00e2/\u00ee) duz sesliyle ayni
    # kavramda topla. Kok neden: CountVectorizer(vocabulary=...) tam-token
    # eslesir ama strip_accents=None (varsayilan), yani "milli" ile "mill\u00ee"
    # ayri sayilir. MHP metni agirlikli "mill\u00ee" yaziyor (TF-IDF'te en
    # yuksek skorlu kelime, 0.270), AKP "milli" yaziyor - normalize
    # etmeden MHP'nin gercek frekansi ~24 kat dusuk gorunuyordu.
    df_ham["milli"] = df_ham["milli"] + df_ham["mill\u00ee"]
    df_ham["adalet"] = df_ham["adalet"] + df_ham["ad\u00e2let"]

    # Sadece nihai kavramlari sec (gosterilecek sirada)
    ham_frekans = df_ham[kavramlar].values

    # Oransal frekans: (ham_sayi / toplam_token) * 10000
    oransal_frekans = np.zeros_like(ham_frekans, dtype=float)
    for i, toplam in enumerate(token_sayilari):
        if toplam > 0:
            oransal_frekans[i] = (ham_frekans[i] / toplam) * 10000

    oransal_df = pd.DataFrame(
        np.round(oransal_frekans, 2),
        index=etiketler,
        columns=kavramlar
    )

    # Konsola yazdir
    print("  " + "=" * 60)
    print("  KAVRAM ORANSAL FREKANSLARI (Her 10.000 Kelimede)")
    print("  " + "=" * 60)

    # Token sayilarini da goster
    for i, etiket in enumerate(etiketler):
        print("    {0}: {1:,} token".format(etiket, token_sayilari[i]))
    print()
    print(oransal_df.to_string())
    print()

    # --- Radar Grafik ---
    kavram_sayisi = len(kavramlar)
    # Acilar (esit aralikli)
    acilar = np.linspace(0, 2 * np.pi, kavram_sayisi, endpoint=False).tolist()
    # Cokgeni kapatmak icin ilk aciyi sona ekle
    acilar += acilar[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for idx, etiket in enumerate(etiketler):
        degerler = oransal_frekans[idx].tolist()
        # Cokgeni kapat (ilk degeri sona ekle)
        degerler += degerler[:1]
        renk = PARTI_RENKLERI.get(etiket, "#333333")

        ax.plot(acilar, degerler, color=renk, linewidth=1.5, marker='o', markersize=5, label=etiket)
        ax.fill(acilar, degerler, color=renk, alpha=0.15)

    # Eksen etiketleri (kavram isimleri)
    ax.set_xticks(acilar[:-1])
    ax.set_xticklabels(kavramlar, fontsize=11, fontweight="medium")

    # Radyal grid cizgileri
    ax.yaxis.set_tick_params(labelsize=9)
    ax.grid(color="#CCCCCC", linestyle="-", linewidth=0.5, alpha=0.7)

    # Baslik
    ax.set_title(
        "Kavram Radar Grafiği\n(10.000 Kelime Başına Frekans - Oransal)",
        fontsize=14, fontweight="bold", pad=30
    )

    # Lejant
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.25, 1.10),
        fontsize=11,
        framealpha=0.9,
        edgecolor="#CCCCCC"
    )

    plt.tight_layout()
    plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    log("BASARILI", "Radar grafigi kaydedildi: {0}".format(cikti_yolu))


# ============================================================================
# C) KAVRAM CUBUK GRAFIGI (GROUPED BAR CHART)
# ============================================================================

def kavram_cubuk_grafigi(corpus, kavramlar, cikti_yolu):
    """
    Belirli kavramlarin parti bazli oransal frekanslarini (her 10.000
    kelimede kullanim sikligi) hesaplayarak gruplu cubuk grafik cizer.

    Radar grafigindeki birebir ayni normalizasyon ve bigram
    birlestirme mantigi kullanilir.
    """
    etiketler = list(corpus.keys())
    belgeler = [corpus[e] for e in etiketler]

    # Her partinin toplam token sayisi
    token_sayilari = [len(corpus[e].split()) for e in etiketler]

    # Genisletilmis sozluk (unigram + bigram)
    arama_kavramlari = [
        "teknoloji", "dijital", "yapay zeka", "yapay zek\u00e2",
        "g\u00fcvenlik", "milli", "mill\u00ee", "sosyal", "liyakat", "adalet", "ad\u00e2let"
    ]

    vectorizer = CountVectorizer(vocabulary=arama_kavramlari, ngram_range=(1, 2))
    count_matris = vectorizer.fit_transform(belgeler)
    ham_frekans_tum = count_matris.toarray()

    df_ham = pd.DataFrame(ham_frekans_tum, columns=arama_kavramlari)
    df_ham["yapay zek\u00e2"] = df_ham["yapay zeka"] + df_ham["yapay zek\u00e2"]
    # bkz. kavram_radar_grafigi() - ayni diyakritik normalizasyonu (\u00e2/\u00ee)
    df_ham["milli"] = df_ham["milli"] + df_ham["mill\u00ee"]
    df_ham["adalet"] = df_ham["adalet"] + df_ham["ad\u00e2let"]
    ham_frekans = df_ham[kavramlar].values

    # Oransal frekans: (ham_sayi / toplam_token) * 10000
    oransal_frekans = np.zeros_like(ham_frekans, dtype=float)
    for i, toplam in enumerate(token_sayilari):
        if toplam > 0:
            oransal_frekans[i] = (ham_frekans[i] / toplam) * 10000

    oransal_df = pd.DataFrame(
        np.round(oransal_frekans, 2),
        index=etiketler,
        columns=kavramlar
    )

    # Seaborn icin long-format DataFrame olustur
    kayitlar = []
    for etiket in etiketler:
        for kavram in kavramlar:
            kayitlar.append({
                "Parti": etiket,
                "Kavram": kavram,
                "Frekans": oransal_df.loc[etiket, kavram]
            })
    df_long = pd.DataFrame(kayitlar)

    # Renk sirasi (etiket sirasina gore)
    renk_listesi = [PARTI_RENKLERI.get(e, "#333333") for e in etiketler]

    # --- Grafik ---
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sns.barplot(
        data=df_long,
        x="Kavram",
        y="Frekans",
        hue="Parti",
        hue_order=etiketler,
        palette=renk_listesi,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )

    ax.set_title(
        "Kavram Frekanslar\u0131 \u2014 Grupland\u0131r\u0131lm\u0131\u015f \u00c7ubuk Grafik\n"
        "(10.000 Kelime Ba\u015f\u0131na D\u00fc\u015fen Oransal Frekans)",
        fontsize=15, fontweight="bold", pad=20
    )
    ax.set_xlabel("Kavram", fontsize=12, labelpad=10)
    ax.set_ylabel("Frekans (10.000 Kelime Ba\u015f\u0131na)", fontsize=12, labelpad=10)
    ax.tick_params(axis="x", labelsize=11, rotation=45)
    ax.tick_params(axis="y", labelsize=10)

    # Lejant
    ax.legend(
        title="Parti",
        fontsize=10,
        title_fontsize=11,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        loc="upper right",
    )

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    log("BASARILI", "Cubuk grafigi kaydedildi: {0}".format(cikti_yolu))


# ============================================================================
# C2) TEMATIK KELIME AILESI / COMPOSITE SKOR
# ============================================================================

def tematik_composite_skor(corpus, cikti_csv_yolu):
    """Iliskili teknoloji kavram ailesini tek bilesik skorda birlestirir.

    Amac: tekil "teknoloji" kelimesine dayali bulgunun, kavram ailesinin
    tamami (teknoloji, dijital, yapay zeka, veri, algoritma, siber) ile
    tutarli olup olmadigini sinamak. Skorlar, belge uzunlugu sapmasini
    gidermek icin 10.000 token basina normalize edilir (radar/cubuk
    grafiklerle ayni normalizasyon).

    Cikti: parti bazli kavram frekanslari, composite ham/normalize skor
    ve tekil "teknoloji" ile composite siralamalarinin karsilastirmasi.
    CSV olarak da kaydedilir.
    """
    etiketler = list(corpus.keys())
    belgeler = [corpus[e] for e in etiketler]
    token_sayilari = [len(corpus[e].split()) for e in etiketler]

    arama_kavramlari = COMPOSITE_TEMA_UNIGRAMLAR + COMPOSITE_TEMA_BIGRAMLAR
    vectorizer = CountVectorizer(vocabulary=arama_kavramlari, ngram_range=(1, 2))
    matris = vectorizer.fit_transform(belgeler).toarray()

    df = pd.DataFrame(matris, columns=arama_kavramlari, index=etiketler)

    # Imla varyantlarini ("yapay zeka" / "yapay zekâ") tek kavramda topla
    df["yapay zekâ (birlesik)"] = df["yapay zeka"] + df["yapay zekâ"]
    df = df.drop(columns=["yapay zeka", "yapay zekâ"])

    kavram_sutunlari = COMPOSITE_TEMA_UNIGRAMLAR + ["yapay zekâ (birlesik)"]
    df["composite_ham"] = df[kavram_sutunlari].sum(axis=1)
    df["toplam_token"] = token_sayilari
    df["composite_10k"] = (df["composite_ham"] / df["toplam_token"] * 10000).round(2)
    df["teknoloji_10k"] = (df["teknoloji"] / df["toplam_token"] * 10000).round(2)

    # Parti siralamasi: tekil kavram vs kavram ailesi (1 = en yuksek)
    df["sira_teknoloji"] = df["teknoloji_10k"].rank(ascending=False).astype(int)
    df["sira_composite"] = df["composite_10k"].rank(ascending=False).astype(int)

    df.index.name = "Parti"
    df.to_csv(cikti_csv_yolu, encoding="utf-8-sig")

    print()
    print("  " + "=" * 60)
    print("  TEMATIK KELIME AILESI / COMPOSITE SKOR (10.000 Token Basina)")
    print("  " + "=" * 60)
    print(df.to_string())
    print()
    if (df["sira_teknoloji"] == df["sira_composite"]).all():
        print("  Siralama karsilastirmasi: tekil 'teknoloji' ile composite")
        print("  skor ayni parti siralamasini veriyor (bulgu tutarli).")
    else:
        farkli = df[df["sira_teknoloji"] != df["sira_composite"]]
        print("  Siralama karsilastirmasi: su partilerde siralama DEGISIYOR:")
        for parti, satir in farkli.iterrows():
            print("    {0}: teknoloji sirasi {1} -> composite sirasi {2}".format(
                parti, satir["sira_teknoloji"], satir["sira_composite"]))
    print()

    log("BASARILI", "Composite skor tablosu kaydedildi: {0}".format(cikti_csv_yolu))
    return df


# ============================================================================
# D) LEXICON TABANLI EKSEN ANALIZI (INOVASYON vs. REGULASYON)
# ============================================================================

def lexicon_eksen_haritasi(corpus, inovasyon_kelimeler, regulasyon_kelimeler, cikti_yolu):
    """
    Sozluk tabanli (lexicon-based) iki boyutlu eksen analizi.
    X ekseni: Etik ve Duzenleme Skoru (Regulasyon sozlugu toplam frekansi)
    Y ekseni: Inovasyon ve Ekonomi Skoru (Inovasyon sozlugu toplam frekansi)
    Degerler 10.000 kelime basina oransal frekans olarak normalize edilir.
    """
    etiketler = list(corpus.keys())
    belgeler = [corpus[e] for e in etiketler]
    token_sayilari = [len(corpus[e].split()) for e in etiketler]

    # Inovasyon skoru: sozlukteki tum kelimelerin toplam frekansi
    vec_ino = CountVectorizer(vocabulary=inovasyon_kelimeler)
    ham_ino = vec_ino.fit_transform(belgeler).toarray().sum(axis=1)

    # Regulasyon skoru: sozlukteki tum kelimelerin toplam frekansi
    vec_reg = CountVectorizer(vocabulary=regulasyon_kelimeler)
    ham_reg = vec_reg.fit_transform(belgeler).toarray().sum(axis=1)

    # Oransal frekans (10.000 kelime basina)
    ino_skorlar = np.array([
        (ham_ino[i] / token_sayilari[i]) * 10000 if token_sayilari[i] > 0 else 0
        for i in range(len(etiketler))
    ])
    reg_skorlar = np.array([
        (ham_reg[i] / token_sayilari[i]) * 10000 if token_sayilari[i] > 0 else 0
        for i in range(len(etiketler))
    ])

    # Konsola yazdir
    print()
    print("  " + "=" * 60)
    print("  LEXICON EKSEN ANALIZI (10.000 Kelime Basina)")
    print("  " + "=" * 60)
    print("  {0:<18} {1:>16} {2:>16}".format("Parti", "Inovasyon (Y)", "Regulasyon (X)"))
    print("  " + "-" * 50)
    for i, etiket in enumerate(etiketler):
        print("  {0:<18} {1:>16.2f} {2:>16.2f}".format(
            etiket, ino_skorlar[i], reg_skorlar[i]))
    print()

    # --- Scatter Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for idx, etiket in enumerate(etiketler):
        x = reg_skorlar[idx]
        y = ino_skorlar[idx]
        renk = PARTI_RENKLERI.get(etiket, "#333333")

        ax.scatter(x, y, color=renk, s=200, zorder=5,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(
            etiket, (x, y),
            textcoords="offset points", xytext=(12, 8),
            fontsize=12, fontweight="bold", color=renk, ha="left",
        )

    # Ceyrek bolme cizgileri (merkezden)
    x_merkez = (reg_skorlar.max() + reg_skorlar.min()) / 2
    y_merkez = (ino_skorlar.max() + ino_skorlar.min()) / 2
    ax.axhline(y=y_merkez, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(x=x_merkez, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_title(
        "\u0130novasyon vs. Reg\u00fclasyon Eksen Haritas\u0131\n"
        "(S\u00f6zl\u00fck Tabanl\u0131 \u2014 10.000 Kelime Ba\u015f\u0131na Oransal Frekans)",
        fontsize=14, fontweight="bold", pad=20
    )
    ax.set_xlabel(
        "Etik ve D\u00fczenleme Skoru (Reg\u00fclasyon Ekseni) \u2192",
        fontsize=12, labelpad=10
    )
    ax.set_ylabel(
        "\u0130novasyon ve Ekonomi Skoru (B\u00fcy\u00fcme Ekseni) \u2192",
        fontsize=12, labelpad=10
    )

    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    log("BASARILI", "Lexicon eksen haritasi kaydedildi: {0}".format(cikti_yolu))


# ============================================================================
# E) TEKNOLOJI BAGLAMLI DUYGU ANALIZI
# ============================================================================

def teknoloji_duygu_analizi(cikti_yolu):
    """
    Orijinal PDF dosyalarini okur, icerisindeki cumleleri ayirir,
    hedef teknoloji kelimelerini barindiran cumleleri tespit eder ve
    lexicon tabanli duygu analizi uygulayarak %100 Yigilmis Cubuk Grafik (Stacked Bar) cizer.
    """
    try:
        import fitz
    except ImportError:
        log("HATA", "PyMuPDF (fitz) yüklü değil. Duygu analizi için fitz gereklidir.")
        return

    beyannameler_klasoru = os.path.join(PROJE_KOKU, "beyannameler")
    beyanname_gruplari = DUYGU_BEYANNAME_GRUPLARI
    hedef_kelimeler = DUYGU_HEDEF_KELIMELER
    pozitif_sozluk = DUYGU_POZITIF_SOZLUK
    negatif_sozluk = DUYGU_NEGATIF_SOZLUK

    # Kelime sinirli (word-boundary) eslestirme desenleri.
    # Onceki surumde alt-dizi (substring) eslestirmesi kullaniliyordu; bu,
    # "guc" -> "guclu(k)", "acik" -> "aciklama" gibi yanlis pozitiflere yol
    # aciyordu. \b...\b ile yalnizca tam kelime eslesir (precision oncelikli).
    poz_desenler = [re.compile(r"\b" + re.escape(k) + r"\b") for k in pozitif_sozluk]
    neg_desenler = [re.compile(r"\b" + re.escape(k) + r"\b") for k in negatif_sozluk]

    sonuclar = []
    cumle_kayitlari = []  # Cumle duzeyinde kayit (dogrulama + seffaflik icin)

    for etiket, dosyalar in beyanname_gruplari.items():
        metin_parcalari = []
        for dosya in dosyalar:
            dosya_yolu = os.path.join(beyannameler_klasoru, dosya)
            if os.path.exists(dosya_yolu):
                try:
                    doc = fitz.open(dosya_yolu)
                    for sayfa in doc:
                        metin_parcalari.append(sayfa.get_text())
                    doc.close()
                except Exception as e:
                    log("UYARI", "{0} okunamadi: {1}".format(dosya, str(e)))

        ham_metin = " ".join(metin_parcalari)
        # PDF'lerdeki satir sonlarini temizle ve cumlelere ayir
        ham_metin = ham_metin.replace("\n", " ")
        cumleler = re.split(r'(?<=[.!?])\s+', ham_metin)

        # Hedef kelimeleri iceren cumleleri filtrele.
        # Orijinal cumle de saklanir: elle kodlama dosyasinda okunabilirlik
        # icin orijinal hali, siniflama icin kucuk harfli hali kullanilir.
        teknoloji_cumleleri = []
        for c in cumleler:
            c_lower = c.replace("I", "ı").replace("İ", "i").lower()
            if any(hk in c_lower for hk in hedef_kelimeler):
                teknoloji_cumleleri.append((c.strip(), c_lower))

        if not teknoloji_cumleleri:
            sonuclar.append({"Parti": etiket, "Pozitif": 0, "Nötr": 100, "Negatif": 0, "Toplam": 0})
            continue

        pozitif_sayisi = 0
        negatif_sayisi = 0
        notr_sayisi = 0

        for c_orijinal, c in teknoloji_cumleleri:
            poz_skor = sum(1 for d in poz_desenler if d.search(c))
            neg_skor = sum(1 for d in neg_desenler if d.search(c))

            if poz_skor > neg_skor:
                algoritma_kod = "pozitif"
                pozitif_sayisi += 1
            elif neg_skor > poz_skor:
                algoritma_kod = "negatif"
                negatif_sayisi += 1
            else:
                algoritma_kod = "nötr"
                notr_sayisi += 1

            cumle_kayitlari.append({
                "Parti": etiket,
                "Cumle": c_orijinal,
                "PozSkor": poz_skor,
                "NegSkor": neg_skor,
                "algoritma_kod": algoritma_kod,
            })

        toplam = len(teknoloji_cumleleri)
        poz_oran = (pozitif_sayisi / toplam) * 100
        notr_oran = (notr_sayisi / toplam) * 100
        neg_oran = (negatif_sayisi / toplam) * 100

        sonuclar.append({
            "Parti": etiket,
            "Pozitif": poz_oran,
            "Nötr": notr_oran,
            "Negatif": neg_oran,
            "Toplam": toplam
        })

    # -- CUMLE DUZEYI VERI SETI VE ELLE KODLAMA ORNEKLEMI (dogrulama) --
    # (1) Tum siniflandirilmis cumleler CSV olarak kaydedilir (seffaflik:
    #     hicbir cumle sessizce elenmez, tum ara sonuclar denetlenebilir).
    # (2) Sozluk tabanli algoritmanin dogrulugunu olcmek icin sabit seedle
    #     rastgele DOGRULAMA_ORNEKLEM_N cumlelik bir orneklem cekilir;
    #     arastirmaci "elle_kod" sutununu doldurur, ardindan
    #     duygu_dogrulama.py ile uyum orani hesaplanir.
    df_cumleler = pd.DataFrame(cumle_kayitlari)
    if not df_cumleler.empty:
        cumle_csv = os.path.join(SONUCLAR_KLASORU, "duygu_cumle_veriseti.csv")
        df_cumleler.to_csv(cumle_csv, index=False, encoding="utf-8-sig")
        log("BASARILI", "Cumle duzeyi duygu veri seti kaydedildi: {0} ({1} cumle)".format(
            cumle_csv, len(df_cumleler)))

        import random
        random.seed(DOGRULAMA_RANDOM_STATE)  # sabit seed - tekrarlanabilirlik icin ZORUNLU
        orneklem_n = min(DOGRULAMA_ORNEKLEM_N, len(df_cumleler))
        if orneklem_n < DOGRULAMA_ORNEKLEM_N:
            log("UYARI", "Toplam cumle sayisi {0} < {1}; orneklem kucultuldu.".format(
                len(df_cumleler), DOGRULAMA_ORNEKLEM_N))
        # random_state=42: tekrarlanabilirlik icin sabitlendi
        orneklem = df_cumleler.sample(n=orneklem_n,
                                      random_state=DOGRULAMA_RANDOM_STATE).copy()
        orneklem.insert(0, "CumleID", orneklem.index)
        orneklem["elle_kod"] = ""  # arastirmaci dolduracak: pozitif/negatif/nötr
        orneklem_csv = os.path.join(SONUCLAR_KLASORU, "elle_kodlama_icin.csv")
        orneklem.to_csv(orneklem_csv, index=False, encoding="utf-8-sig")
        log("BASARILI", "Elle kodlama orneklemi kaydedildi: {0} (n={1}, seed={2})".format(
            orneklem_csv, orneklem_n, DOGRULAMA_RANDOM_STATE))

    # -- DUSUK ORNEKLEM (n < MIN_GUVENILIR_N) KONTROLU --
    # Oranlarin istatistiksel guvenilirligi icin otomatik esik kontrolu.
    dusuk_n_partiler = [(r["Parti"], r["Toplam"]) for r in sonuclar
                        if r["Toplam"] < MIN_GUVENILIR_N]
    for parti, n in dusuk_n_partiler:
        print("  UYARI: {0} için n={1} (< {2}), oranlar istatistiksel "
              "olarak güvenilir değildir.".format(parti, n, MIN_GUVENILIR_N))

    # -- OZET SONUC TABLOSUNU CSV OLARAK KAYDET (guvenilirlik bayragiyla) --
    df_duygu = pd.DataFrame(sonuclar)
    df_duygu["Guvenilir (n>={0})".format(MIN_GUVENILIR_N)] = (
        df_duygu["Toplam"] >= MIN_GUVENILIR_N)
    duygu_csv = os.path.join(SONUCLAR_KLASORU, "duygu_analizi_sonuclari.csv")
    df_duygu.to_csv(duygu_csv, index=False, encoding="utf-8-sig")
    log("BASARILI", "Duygu analizi sonuc tablosu kaydedildi: {0}".format(duygu_csv))

    # Terminale yazdir
    print()
    print("  " + "=" * 60)
    print("  TEKNOLOJI ODAKLI DUYGU ANALIZI (% ORANLARI)")
    print("  " + "=" * 60)
    for res in sonuclar:
        print("  {0:<15}: %{1:>5.1f} Pozitif | %{2:>5.1f} Nötr | %{3:>5.1f} Negatif  (Toplam: {4})".format(
            res["Parti"], res["Pozitif"], res["Nötr"], res["Negatif"], res["Toplam"]
        ))
    print()

    # --- Yuzde Yigilmis Cubuk Grafik (100% Stacked Bar Chart) ---
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    partiler = [r["Parti"] for r in sonuclar]
    pozitifler = [r["Pozitif"] for r in sonuclar]
    notrler = [r["Nötr"] for r in sonuclar]
    negatifler = [r["Negatif"] for r in sonuclar]
    
    bar_width = 0.65
    
    # Asagidan yukariya: Pozitif, Notr, Negatif (birlesik DUYGU_RENKLERI paleti)
    p1 = ax.bar(partiler, pozitifler, bar_width, color=DUYGU_RENKLERI["Pozitif"], edgecolor="white", linewidth=1.2, label="Pozitif")
    p2 = ax.bar(partiler, notrler, bar_width, bottom=pozitifler, color=DUYGU_RENKLERI["Nötr"], edgecolor="white", linewidth=1.2, label="Nötr")

    bottom_negatif = [pozitifler[i] + notrler[i] for i in range(len(partiler))]
    p3 = ax.bar(partiler, negatifler, bar_width, bottom=bottom_negatif, color=DUYGU_RENKLERI["Negatif"], edgecolor="white", linewidth=1.2, label="Negatif")

    # Yuzde etiketlerini bar uzerine yazdir (Kalin ve renklendirilmis yazilar)
    renk_eslesmesi = [(p1, "white"), (p2, RENK_METIN), (p3, "white")]
    for bars, text_color in renk_eslesmesi:
        for bar in bars:
            height = bar.get_height()
            if height > 3: # 3%'den kucukse yazi sigmayabilir, goze hitap etmesi icin yazilmaz
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + height / 2,
                    "%{0:.1f}".format(height),
                    ha="center", va="center", color=text_color, fontweight="bold", fontsize=11
                )
                
    ax.set_title(
        "Teknoloji Odaklı Duygu Analizi\n"
        "(Pozitif, Nötr ve Negatif Bağlamsal Oranlar)",
        fontsize=15, fontweight="bold", pad=40
    )
    ax.set_ylabel("Oran (%)", fontsize=13, labelpad=10)
    ax.set_ylim(0, 100)
    
    # Arka Plan ve Eksenler (Despining)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Sadece Y ekseni icin arkadan gecen kesik cizgili hafif izgara
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.grid(False)
    
    # Lejant siralmasi (Negatif ustte, Nötr ortada, Pozitif altta olacak sekilde)
    handles, labels = ax.get_legend_handles_labels()
    # Lejantı başlığın altına, yatay (ncol=3) ve çerçevesiz olarak ekle
    ax.legend(handles[::-1], labels[::-1], loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=11)
    
    # Eksen etiketleri boyutlandirmasi
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)

    # Dusuk orneklem dipnotu: uyari grafik dosyasinin altina otomatik yazilir
    if dusuk_n_partiler:
        dipnot = "Uyarı: " + ", ".join(
            "{0} (n={1})".format(p, n) for p, n in dusuk_n_partiler)
        dipnot += " için n < {0}; oranlar istatistiksel olarak güvenilir değildir.".format(
            MIN_GUVENILIR_N)
        fig.text(0.5, -0.02, dipnot, ha="center", va="top",
                 fontsize=9, color="#555555", style="italic")

    plt.tight_layout()
    plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    log("BASARILI", "Duygu analizi grafigi kaydedildi: {0}".format(cikti_yolu))

    return sonuclar


# ============================================================================
# F) TEKNOLOJI ODAKLI SOYLEM AKIS HARITASI (SANKEY)
# ============================================================================

def teknoloji_sankey_diyagrami(sonuclar, cikti_yolu):
    """
    Duygu analizi fonksiyonundan gelen dinamik sonuclari (sonuclar) alarak,
    partilerden duygu durumlarina dogru akan bir Sankey diyagrami cizer
    ve yuksek cozunurluklu (300 DPI) statik PNG olarak kaydeder.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        log("HATA", "Plotly yüklü değil. Sankey için 'pip install plotly kaleido' gerekli.")
        return
        
    # Etiketler (Sol taraf: Partiler, Sag taraf: Duygular)
    labels = ["AKP", "CHP_İYİ_Ortak", "DEM Parti", "MHP", "Pozitif", "Nötr", "Negatif"]
    
    # Hex renk kodlarindan RGBA ureten yardimci fonksiyon
    def hex_to_rgba(hex_code, opacity=1.0):
        hex_code = hex_code.lstrip('#')
        rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"
        
    # Node Renkleri: diger tum grafiklerle ayni birlesik paletten alinir.
    node_colors_hex = [
        PARTI_RENKLERI["AKP"],            # 0: AKP
        PARTI_RENKLERI["CHP_IYI_Ortak"],  # 1: CHP_IYI_Ortak
        PARTI_RENKLERI["DEM"],            # 2: DEM Parti
        PARTI_RENKLERI["MHP"],            # 3: MHP
        DUYGU_RENKLERI["Pozitif"],        # 4: Pozitif
        DUYGU_RENKLERI["Nötr"],           # 5: Nötr
        DUYGU_RENKLERI["Negatif"],        # 6: Negatif
    ]
    node_colors = [hex_to_rgba(c, 0.9) for c in node_colors_hex]
    
    source = []
    target = []
    value = []
    link_colors = []
    
    # Dinamik olarak sonuclardan oku ve baglantilari (link) olustur
    # sonuclar list of dict: {"Parti": "AKP", "Pozitif": 69.0, "Nötr": 28.4, "Negatif": 2.6}
    party_to_source = {"AKP": 0, "CHP_IYI_Ortak": 1, "DEM": 2, "MHP": 3}
    
    for r in sonuclar:
        p_name = r.get("Parti")
        if p_name not in party_to_source:
            continue
            
        s_idx = party_to_source[p_name]
        
        # Link rengi, kaynak node'un rengini alacak (opacity=0.45)
        flow_color = hex_to_rgba(node_colors_hex[s_idx], 0.45)
        
        # Pozitif akis (target = 4)
        if r["Pozitif"] > 0:
            source.append(s_idx)
            target.append(4)
            value.append(r["Pozitif"])
            link_colors.append(flow_color)
            
        # Nötr akis (target = 5)
        if r["Nötr"] > 0:
            source.append(s_idx)
            target.append(5)
            value.append(r["Nötr"])
            link_colors.append(flow_color)
            
        # Negatif akis (target = 6)
        if r["Negatif"] > 0:
            source.append(s_idx)
            target.append(6)
            value.append(r["Negatif"])
            link_colors.append(flow_color)

    # 1. Dinamik Etiketler ve Koordinat Hesaplamalari
    T = sum(value)
    if T == 0:
        log("UYARI", "Sankey cizimi icin hic veri yok.")
        return
        
    active_sources = [s for s in range(4) if sum(v for src, t, v in zip(source, target, value) if src == s) > 0]
    active_targets = [t for t in range(4, 7) if sum(v for s, tgt, v in zip(source, target, value) if tgt == t) > 0]
    
    pad_fraction = 0.05
    usable_height_source = 1.0 - (len(active_sources) - 1) * pad_fraction if len(active_sources) > 0 else 1.0
    usable_height_target = 1.0 - (len(active_targets) - 1) * pad_fraction if len(active_targets) > 0 else 1.0
    
    node_y = [0.0] * 7
    link_coords = {}
    
    current_y = 0.0
    for s_idx in active_sources:
        links_for_s = [(i, t, val) for i, (s, t, val) in enumerate(zip(source, target, value)) if s == s_idx]
        links_for_s.sort(key=lambda item: item[1])
        
        s_val = sum(val for _, _, val in links_for_s)
        h = (s_val / T) * usable_height_source
        node_y[s_idx] = current_y + h / 2
        
        temp_y = current_y
        for link_idx, t, val in links_for_s:
            link_h = (val / T) * usable_height_source
            link_coords[link_idx] = {'start_y': temp_y + link_h / 2}
            temp_y += link_h
            
        current_y += h + pad_fraction

    current_y = 0.0
    for t_idx in active_targets:
        links_for_t = [(i, s, val) for i, (s, t, val) in enumerate(zip(source, target, value)) if t == t_idx]
        links_for_t.sort(key=lambda item: item[1])
        
        t_val = sum(val for _, _, val in links_for_t)
        h = (t_val / T) * usable_height_target
        node_y[t_idx] = current_y + h / 2
        
        temp_y = current_y
        for link_idx, s, val in links_for_t:
            link_h = (val / T) * usable_height_target
            link_coords[link_idx]['end_y'] = temp_y + link_h / 2
            temp_y += link_h
            
        current_y += h + pad_fraction

    # 2. Etiketleri Guncelle
    for t_idx in active_targets:
        t_val = sum(v for s, t, v in zip(source, target, value) if t == t_idx)
        avg_pct = (t_val / T) * 100
        labels[t_idx] = f"{labels[t_idx]}<br>(%{avg_pct:.1f})"

    for r in sonuclar:
        p_name = r.get("Parti")
        if p_name in party_to_source:
            s_idx = party_to_source[p_name]
            toplam_cumle = r.get("Toplam", 0)
            labels[s_idx] = f"{labels[s_idx]}<br>({toplam_cumle} Cümle)"

    # 3. Annotationlari (Yuzdeleri) Olustur
    annotations = []
    
    # Target index'e gore yatayda (t) dagit: 4(Poz): 0.3, 5(Notr): 0.5, 6(Neg): 0.7
    # Boylece etiketler ust uste binmez, farkli sutunlarda gosterilir.
    t_values_for_target = {4: 0.30, 5: 0.50, 6: 0.70}
    
    for i in range(len(value)):
        sy = link_coords[i]['start_y']
        ey = link_coords[i]['end_y']
        val = value[i]
        s_idx = source[i]
        t_idx = target[i]
        
        # Akisin uzerindeki egrisel konumu (t parametresi)
        t = t_values_for_target.get(t_idx, 0.5)
        
        # Bezier egrisi (Sankey linkleri icin cubic bezier formulu)
        w1 = ((1 - t) ** 3) + 3 * ((1 - t) ** 2) * t
        w2 = 3 * (1 - t) * (t ** 2) + (t ** 3)
        curve_y = w1 * sy + w2 * ey
        
        curve_x = 1.5 * ((1 - t) ** 2) * t + 1.5 * (1 - t) * (t ** 2) + (t ** 3)
        
        # Plotly paper Y koordinati
        paper_y = 1.0 - curve_y
        
        # Arka plan rengini parti renginden (source_color) al
        src_hex = node_colors_hex[s_idx]
        bg_color = hex_to_rgba(src_hex, 0.85)
        
        annotations.append(go.layout.Annotation(
            x=curve_x,
            y=paper_y,
            xref="paper",
            yref="paper",
            text=f"%{val:.1f}",
            showarrow=False,
            font=dict(size=12, color="white", family="Arial", weight="bold"),
            bgcolor=bg_color,
            bordercolor='rgba(255,255,255,0.4)',
            borderwidth=1,
            borderpad=3
        ))

    # Dusuk orneklem dipnotu: uyari Sankey diyagraminin altina otomatik yazilir
    dusuk_n_partiler = [(r.get("Parti"), r.get("Toplam", 0)) for r in sonuclar
                        if r.get("Toplam", 0) < MIN_GUVENILIR_N]
    if dusuk_n_partiler:
        dipnot = "Uyarı: " + ", ".join(
            "{0} (n={1})".format(p, n) for p, n in dusuk_n_partiler)
        dipnot += " için n < {0}; oranlar istatistiksel olarak güvenilir değildir.".format(
            MIN_GUVENILIR_N)
        annotations.append(go.layout.Annotation(
            x=0.5, y=-0.07, xref="paper", yref="paper", text=dipnot,
            showarrow=False, font=dict(size=12, color="#555555")))

    # Figuru olustur
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="white", width=1),
            label=labels,
            color=node_colors,
            x=[0.001]*4 + [0.999]*3,
            y=node_y
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    )])
    
    fig.update_layout(
        title_text="Şekil 5. Teknoloji Odaklı Söylem Akış Haritası (Sankey Diyagramı)",
        title_font_size=18,
        title_x=0.5,
        font=dict(size=14, color="#333333", family="Arial"),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1000,
        height=650,
        margin=dict(t=80, b=70, l=40, r=40),
        annotations=annotations
    )
    
    try:
        # scale=3 -> yuksek cozunurluk (300 DPI karsiligi)
        fig.write_image(cikti_yolu, scale=3)
        log("BASARILI", "Sankey diyagrami dinamik verilerle olusturuldu: {0}".format(cikti_yolu))
    except Exception as e:
        log("HATA", "Sankey kaydedilemedi (Kaleido yüklü olmayabilir veya bir hata oluştu): " + str(e))


# ============================================================================
# ANA IS AKISI
# ============================================================================

def main():
    """Ana fonksiyon: JSON corpus'u yukle, gorsellestirmeleri uret."""
    print()
    print("=" * 60)
    print("  BEYANNAME GORSELLESTIRME MODULU")
    print("=" * 60)
    print()

    # Tum grafikler icin tutarli akademik stili uygula
    akademik_stil_uygula()

    # Kanonik corpusu yukle (tek dogruluk kaynagi)
    corpus, corpus_meta = corpus_yukle(SONUCLAR_KLASORU)
    if corpus is None:
        return

    # Cikti klasorunu olustur
    os.makedirs(SONUCLAR_KLASORU, exist_ok=True)

    # A) Ideolojik Uzay Haritasi (PCA)
    pca_yolu = os.path.join(SONUCLAR_KLASORU, "pca_ideolojik_uzay.png")
    log("BILGI", "PCA ideolojik uzay haritasi hazirlaniyor...")
    ideolojik_uzay_haritasi(corpus, pca_yolu)

    # B) Kavram Radar Grafigi
    radar_yolu = os.path.join(SONUCLAR_KLASORU, "radar_kavramlar.png")
    log("BILGI", "Kavram radar grafigi hazirlaniyor...")
    kavram_radar_grafigi(corpus, RADAR_KAVRAMLARI, radar_yolu)

    # C) Kavram Cubuk Grafigi (Grouped Bar Chart)
    bar_yolu = os.path.join(SONUCLAR_KLASORU, "bar_kavramlar.png")
    log("BILGI", "Kavram cubuk grafigi hazirlaniyor...")
    kavram_cubuk_grafigi(corpus, RADAR_KAVRAMLARI, bar_yolu)

    # C2) Tematik Kelime Ailesi / Composite Skor
    composite_yolu = os.path.join(SONUCLAR_KLASORU, "composite_teknoloji_skoru.csv")
    log("BILGI", "Tematik composite skor hesaplaniyor...")
    tematik_composite_skor(corpus, composite_yolu)

    # D) Lexicon Tabanli Eksen Analizi (Inovasyon vs. Regulasyon)
    lexicon_yolu = os.path.join(SONUCLAR_KLASORU, "lexicon_eksen_haritasi.png")
    log("BILGI", "Lexicon eksen haritasi hazirlaniyor...")
    lexicon_eksen_haritasi(corpus, INOVASYON_SOZLUGU, REGULASYON_SOZLUGU, lexicon_yolu)

    # E) Teknoloji Baglamli Duygu Analizi
    duygu_yolu = os.path.join(SONUCLAR_KLASORU, "duygu_analizi_teknoloji_modern.png")
    log("BILGI", "Teknoloji odakli duygu analizi hazirlaniyor...")
    sonuclar = teknoloji_duygu_analizi(duygu_yolu)

    # F) Sankey Diyagrami
    if sonuclar:
        sankey_yolu = os.path.join(SONUCLAR_KLASORU, "duygu_sankey_teknoloji.png")
        log("BILGI", "Sankey diyagrami (Akis Haritasi) hazirlaniyor...")
        teknoloji_sankey_diyagrami(sonuclar, sankey_yolu)

    print("=" * 60)
    print("  GORSELLESTIRME TAMAMLANDI")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
