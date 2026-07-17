# -*- coding: utf-8 -*-
"""
Duygu Sozlugu Dogrulama - Uyum Hesaplama Araci
==============================================
gorsellestirme.py calistirildiginda sonuclar/elle_kodlama_icin.csv uretilir
(sabit seed=42 ile cekilmis dogrulama orneklemi; bkz. DOGRULAMA_RANDOM_STATE).
Arastirmaci bu dosyadaki "elle_kod" sutununu doldurur. Gecerli degerler:
pozitif, negatif, notr (veya nötr). Ardindan bu script calistirilir:

    python duygu_dogrulama.py [csv_yolu]

Cikti: ham uyum orani (percent agreement), Cohen kappa katsayisi ve
karisiklik (confusion) tablosu. Sonuclar konsola yazilir ve
sonuclar/duygu_dogrulama_sonucu.txt dosyasina kaydedilir.
"""

import os
import sys

import pandas as pd

VARSAYILAN_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sonuclar", "elle_kodlama_icin.csv")

GECERLI_KODLAR = {"pozitif", "negatif", "nötr"}


def uyum_hesapla(elle_kodlanan_csv_path):
    """Elle kodlanan CSV ile algoritma kodlari arasindaki uyumu hesaplar.

    Eksik veya gecersiz elle_kod degerleri varsa hesaplama YAPILMAZ;
    sorunlu satirlar acikca raporlanir (sessiz eleme yok).

    Dondurur: (uyum_orani, kappa, karisiklik_tablosu) veya None.
    """
    df = pd.read_csv(elle_kodlanan_csv_path, encoding="utf-8-sig", sep=";")

    if "elle_kod" not in df.columns or "algoritma_kod" not in df.columns:
        print("HATA: CSV icinde elle_kod / algoritma_kod sutunlari bulunamadi.")
        return None

    # NaN satirlari at, sonra her iki sutunu string'e cevirip kirp.
    # Excel'den disari aktarilan CSV'lerde bos hucreler bazen tek bosluk
    # karakteri olarak geliyor, NaN degil - o yuzden strip() sonrasi
    # tekrar bos string kontrolu de gerekiyor.
    df = df.dropna(subset=["elle_kod", "algoritma_kod"])
    df["elle_kod"] = df["elle_kod"].astype(str).str.strip()
    df["algoritma_kod"] = df["algoritma_kod"].astype(str).str.strip()
    df = df[(df["elle_kod"] != "") & (df["algoritma_kod"] != "")]

    # Normalize: kirp, Turkce I/İ kurali ile kucult, notr -> nötr esle
    df["elle_kod"] = (df["elle_kod"]
                      .str.replace("I", "ı").str.replace("İ", "i").str.lower()
                      .replace({"notr": "nötr"}))
    df["algoritma_kod"] = (df["algoritma_kod"]
                           .str.replace("I", "ı").str.replace("İ", "i").str.lower()
                           .replace({"notr": "nötr"}))

    bos_maske = df["elle_kod"].isin(["", "nan", "none"])
    if bos_maske.any():
        kimlikler = (df.loc[bos_maske, "CumleID"].tolist()
                     if "CumleID" in df.columns
                     else df.index[bos_maske].tolist())
        print("HATA: {0} satirda elle_kod bos. Once tum satirlari kodlayin.".format(
            int(bos_maske.sum())))
        print("Bos satirlar (CumleID): {0}".format(kimlikler))
        return None

    gecersiz_maske = ~df["elle_kod"].isin(GECERLI_KODLAR)
    if gecersiz_maske.any():
        print("HATA: Gecersiz elle_kod degerleri: {0}".format(
            sorted(df.loc[gecersiz_maske, "elle_kod"].unique())))
        print("Gecerli degerler: pozitif, negatif, notr (veya nötr)")
        return None

    uyum_orani = float((df["elle_kod"] == df["algoritma_kod"]).mean())

    # Cohen kappa: sans duzeltmeli uyum katsayisi. Yalnizca ham uyum orani
    # sinif dengesizliginde yaniltici olabilecegi icin ek olarak raporlanir.
    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = float(cohen_kappa_score(df["elle_kod"], df["algoritma_kod"]))
    except ImportError:
        kappa = None

    karisiklik = pd.crosstab(df["elle_kod"], df["algoritma_kod"],
                             rownames=["Elle Kod"], colnames=["Algoritma Kod"])

    satirlar = []
    satirlar.append("DUYGU SOZLUGU DOGRULAMA SONUCU")
    satirlar.append("=" * 50)
    satirlar.append("Orneklem buyuklugu : {0}".format(len(df)))
    satirlar.append("Ham uyum orani     : {0:.1%}".format(uyum_orani))
    if kappa is not None:
        satirlar.append("Cohen kappa        : {0:.3f}".format(kappa))
    satirlar.append("")
    satirlar.append("Karisiklik tablosu (satir: elle kod, sutun: algoritma):")
    satirlar.append(karisiklik.to_string())
    rapor = "\n".join(satirlar)
    print(rapor)

    cikti_yolu = os.path.join(
        os.path.dirname(os.path.abspath(elle_kodlanan_csv_path)),
        "duygu_dogrulama_sonucu.txt")
    with open(cikti_yolu, "w", encoding="utf-8") as f:
        f.write(rapor + "\n")
    print()
    print("Sonuc dosyaya da kaydedildi: {0}".format(cikti_yolu))

    return uyum_orani, kappa, karisiklik


if __name__ == "__main__":
    yol = sys.argv[1] if len(sys.argv) > 1 else VARSAYILAN_CSV
    if not os.path.isfile(yol):
        print("HATA: Dosya bulunamadi: {0}".format(yol))
        print("Once gorsellestirme.py calistirilmali (elle_kodlama_icin.csv uretir),")
        print("ardindan elle_kod sutunu elle doldurulmalidir.")
        sys.exit(1)
    uyum_hesapla(yol)
