# 2023 Genel Seçimleri Parti Beyannamelerinde Yapay Zeka Söylemi

Bu depo, "Türkiye'de Yapay Zeka ve Dijitalleşme Söyleminin İdeolojik Çerçevelenmesi:
2023 Genel Seçimleri Parti Beyannameleri Üzerine Bir Metin Madenciliği Analizi" başlıklı
yüksek lisans tezimin analiz kodunu ve buna bağlı tüm ara/nihai çıktıları içerir. Tezin
2.3 (Güvenirlik) bölümünde de belirtildiği gibi, bulguların denetlenebilir ve
tekrarlanabilir olması için kullanılan tüm kodlar, sözlükler ve ön işleme çıktıları
burada eksiksiz paylaşılıyor.

Aşağıdaki metin bir kurulum kılavuzundan çok, depoda neyin nerede olduğunu ve hangi
kararın neden alındığını açıklamak için yazıldı. Yöntemin tam gerekçesi ve akademik
atıfları tezin "Araştırmanın Yöntemi" bölümünde; burada onun kod seviyesindeki karşılığı var.

## Veri seti

Analiz birimi, 2023 Genel Seçimleri'nde yarışan beş partinin resmi seçim beyannameleridir.
Beyannameler `veri/beyannameler/` altında PDF olarak duruyor; hiçbiri özel/gizli bir
kaynaktan gelmiyor, hepsi partilerin kendi yayımladığı kamuya açık belgeler.

AKP'nin iki ayrı PDF halinde yayımladığı beyanname tek metin bloğunda birleştirildi.
CHP ve İYİ Parti, Millet İttifakı çatısı altında ortak bir mutabakat metni yayımladığı
ve bu metin iki partiye ayrıştırılamadığı için tek analiz birimi (`CHP_IYI_Ortak`) olarak
işlendi. Bu yüzden 5 kaynak dosyadan 4 analiz birimi çıkıyor:

| Analiz birimi | Kaynak dosya | Ham kelime | İşlenmiş token | Azalma |
|---|---|---|---|---|
| AKP | 2 | 155.223 | 73.377 | %52,7 |
| CHP_IYI_Ortak | 1 | 52.535 | 29.907 | %43,1 |
| MHP | 1 | 40.291 | 22.618 | %43,9 |
| DEM | 1 | 12.560 | 6.955 | %44,6 |
| **Toplam** | **5** | **260.609** | **132.857** | **%49,0** |

Bu sayılar `sonuclar/betimsel_istatistikler.csv` dosyasından geliyor ve
`beyanname_analiz.py` her çalıştığında yeniden üretiliyor. DEM Parti'nin derlem hacmi
belirgin şekilde küçük olduğu için, tezde de vurgulandığı üzere DEM'e dair oransal
bulgular ihtiyatla okunmalı (özellikle duygu analizinde DEM'in teknoloji bağlamlı cümle
sayısı 3'te kaldığından güvenilirlik uyarısı otomatik ekleniyor, bkz. `MIN_GUVENILIR_N`).

Partilerin ideolojik konumu CSES Türkiye 2023 Makro Raporu'ndaki uzman yerleştirmesine
dayanıyor (0=sol, 10=sağ): AKP 9, MHP 10, İYİ Parti 7, CHP 4, DEM (Yeşil Sol Parti üzerinden) 1.
Üç kategorili sınıflamada AKP+MHP "sağ", CHP_IYI_Ortak ortalama skoru ve tek metin olması
nedeniyle "merkez (karma)", DEM ise "hak temelli sol" olarak kodlanıyor. Bu kodlama
herhangi bir analiz betiğinde kullanılmıyor, sadece bulguların yorumlanmasında (tez
Bölüm 3.3 ve 4. Bölüm) referans alınıyor.

## Pipeline nasıl işliyor

İki bağımsız script var, ikisi de aynı klasördeki dosyaları okuyup `sonuclar/` altına yazıyor.

**`kod/beyanname_analiz.py`** — ağır iş burada. PDF'leri PyMuPDF ile sayfa sayfa okuyor,
sonra üç aşamalı bir onarım/temizlik uyguluyor:

1. Satır sonu heceleme birleştirme ve PDF'e özgü bozuk kalıpların (sayfa üstbilgileri,
   harf aralı dekoratif başlıklar, tekrar eden bağlaç öbekleri) sözlük tabanlı düzeltilmesi
2. Küçük harfe çevirme (Türkçe I/İ sorununu elle çözerek — bkz. aşağıdaki not), URL/rakam/
   noktalama temizliği
3. spaCy'nin `tr_core_news_md` modeliyle lemmatizasyon, ardından yalnızca isim ve sıfat
   etiketli tokenların tutulması, ardından dolgu kelime filtresi

Her filtre adımında kaç token elendiği ayrıca kaydediliyor ve `sonuclar/filtre_dokumu.csv`
olarak dışa aktarılıyor — yani hiçbir kelime sessizce kaybolmuyor, süreç baştan sona
izlenebilir. Script'in ürettiği en önemli dosya `sonuclar/corpus_final_v1_<tarih>.json`:
bundan sonraki her analiz (görselleştirme dahil) yalnızca bu dosyadan besleniyor. `_meta`
bloğunda o çalıştırmada kullanılan kütüphane sürümleri, spaCy pipeline ayarları ve
parti bazlı filtre dökümü saklı, böylece hangi çıktının hangi ortamda üretildiği geriye
dönük olarak da doğrulanabiliyor.

**`kod/gorsellestirme.py`** — kanonik JSON'daki en güncel tarihli dosyayı otomatik
seçip okuyor ve TF-IDF/PCA ile ideolojik uzam haritası, kavram bazlı radar/çubuk grafikleri,
sözlük tabanlı inovasyon-regülasyon ekseni ve teknoloji odaklı duygu analizini (+ Sankey
akış diyagramı) üretiyor. PCA `svd_solver="full"` ile deterministik çalışıyor, yani her
çalıştırmada aynı koordinatları veriyor — n=4 gibi küçük bir örneklemde rastgele bir
solver'ın çalıştırmadan çalıştırmaya farklı sonuç vermesi istenmiyordu.

**`kod/duygu_dogrulama.py`** — sözlük tabanlı duygu sınıflandırmasının ne kadar güvenilir
olduğunu görmek için ayrı bir doğrulama adımı. `gorsellestirme.py` teknoloji bağlamlı
cümlelerden parti hacimleriyle orantılı, `random_state=42` ile sabitlenmiş 60 cümlelik bir
örneklem çekiyor (`sonuclar/elle_kodlama_icin.csv`); ben bu örneklemi algoritma çıktısını
görmeden elle pozitif/nötr/negatif olarak kodladım, script de ham uyum oranını ve Cohen
kappa'yı hesaplıyor. Sonuç: %65,5 ham uyum, kappa=0,382 — orta düzeyin altında bir uyum.
Bunu gizlemiyorum çünkü sözlük tabanlı yöntemin sınırlarını gösteren dürüst bir bulgu;
tezin 2.3 bölümünde de aynı şekilde tartışılıyor. Uyumsuzluğun neredeyse tamamı
nötr-pozitif sınırında yoğunlaşıyor, negatif kategori her iki kodlamada da marjinal kalıyor.

## Türkçeye özgü birkaç ayrıntı

Python'da `"PARTİ".lower()` çağırmak `parti̇` (sonunda gizli bir nokta karakteriyle)
üretiyor, çünkü Python'ın küçültme kuralı İngilizce. Bu da kelimeyi stopword filtresinden
kaçırıyor. Kodda bu yüzden `.lower()` çağrılmadan önce `I`/`İ` harfleri elle Türkçe
karşılıklarına çevriliyor. Küçük bir detay ama atlanırsa özellikle parti isimlerinden
türeyen kelimeler filtreden sızıyor ve TF-IDF sonuçlarını bozuyor.

Lemmatizasyonda da benzer bir sorun var: `tr_core_news_md` modeli şapkalı/şapkasız yazılan
kelimeleri (hak/hâk, hal/hâl gibi) farklı kök olarak algılayabiliyor. Bunu tamamen
otomatik çözmek yerine, regülasyon sözlüğüne kelimenin her iki yazımını da elle ekledim —
tam bir çözüm değil ama en azından bilinen bir sınırlılık olarak dokümante edilmiş oluyor.

## Klasör yapısı

```
kod/                           analiz betikleri
  beyanname_analiz.py          PDF -> onarım -> temizlik -> lemmatizasyon -> POS/stopword filtresi -> istatistik/TF-IDF/bigram
  gorsellestirme.py            kanonik corpus -> PCA, radar, bar, sözlük ekseni, duygu analizi, Sankey
  duygu_dogrulama.py           elle kodlanan örnekle duygu sözlüğünün güvenirlik kontrolü (Cohen kappa)
veri/beyannameler/             ham PDF kaynak belgeler (5 dosya, 4 analiz birimi)
sonuclar/                      her iki scriptin ürettiği tüm CSV/PNG + kanonik JSON derlem
requirements.txt               sabitlenmiş kütüphane sürümleri
```

## Çalıştırma

```bash
pip install -r requirements.txt

# Türkçe spaCy modeli - resmi "spacy download" komutu bende çalışmadı,
# HuggingFace'teki whl'i doğrudan --no-deps ile kurmak gerekiyor
pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_md/resolve/main/tr_core_news_md-1.0-py3-none-any.whl --no-deps

python kod/beyanname_analiz.py     # ~6-8.5 dk, PDF'ler veri/beyannameler/ altında olmalı
python kod/gorsellestirme.py       # birkaç saniye, yukarıdaki adımın ürettiği JSON'a ihtiyaç duyar

# İsteğe bağlı: sonuclar/elle_kodlama_icin.csv içindeki elle_kod sütunu elle doldurulduktan sonra
python kod/duygu_dogrulama.py
```

## Sınırlılıklar

Bunları tezin 2.4 bölümünde de tartışıyorum ama burada da açıkça tekrar etmek istiyorum,
çünkü kodu çalıştırıp sonuçlara bakacak biri bunları bilmeden yanlış çıkarım yapabilir:

- PCA ve TF-IDF karşılaştırmaları yalnızca 4 analiz birimi üzerinden yapılıyor. Bu, istatistiksel
  genelleme için yeterli bir örneklem değil; PCA haritasındaki örüntüler kanıt değil, keşifsel
  bir harita olarak okunmalı.
- Sözlük tabanlı duygu analizi kelimenin yalın haline göre eşleşiyor, olumsuzlama (örn. "güvenli
  değil") kelime kökünün taşıdığı anlama göre değerlendiriliyor — cümle düzeyinde gerçek anlamı
  her zaman yakalayamıyor.
- CHP ve İYİ Parti metinleri tek belge olarak yayımlandığı için bu iki partinin söylemini
  birbirinden ayırmak mümkün değil.
- Dolgu kelime listesindeki ek kelimeler (parti isimlerinden türeyenler ve konu dışı ama sık
  geçen ifadeler) araştırmacı kararına dayanıyor; bu yüzden `beyanname_analiz.py` içindeki
  `OZEL_STOPWORDS` listesi hiçbir şey gizlenmeden olduğu gibi paylaşılıyor.

## Lisans

Kod (`kod/`) ve bu kod tarafından üretilen çıktılar (`sonuclar/`) MIT lisansı altında
(bkz. `LICENSE`). `veri/beyannameler/` altındaki PDF'ler bu lisansın kapsamında değil —
onlar ilgili partilerin resmi olarak yayımladığı belgeler, burada yalnızca tekrarlanabilirlik
amacıyla bulunuyor.
