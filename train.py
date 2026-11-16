folder = Path(_file_).resolve().parent
lokasi_data = folder / "data" / "dataset_sms_spam_v1.csv"

data = pd.read_csv(lokasi_data)

print("Contoh isi dataset:")
print(data.head())
print("\nJumlah data awal:", len(data))
print("\nJumlah data setiap label:")
print(data["label"].value_counts().sort_index())

data = data.dropna(subset=["teks", "label"]).copy()
data["teks"] = data["teks"].str.lower()
data["teks"] = data["teks"].str.replace(
    r"\s+", " ", regex=True
).str.strip()
data = data[data["teks"] != ""]

if not data["label"].isin([0, 1, 2]).all():
    raise ValueError("Label harus berupa 0, 1, atau 2.")

jumlah_label = data.groupby("teks")["label"].nunique()
teks_konflik = jumlah_label[jumlah_label > 1].index

data = data[~data["teks"].isin(teks_konflik)]
data = data.drop_duplicates(subset="teks")

print("\nJumlah data setelah dibersihkan:", len(data))

x = data["teks"]
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nJumlah data latih:", len(x_train))
print("Jumlah data uji:", len(x_test))

model = Pipeline([
  ("tfidf",
        TfidfVectorizer(ngram_range=(1, 2)) ),
    ( "naive_bayes",MultinomialNB(alpha=1.0) )
])
model.fit(x_train, y_train)
print("\nPelatihan model selesai.")

prediksi = model.predict(x_test)
akurasi = accuracy_score(y_test, prediksi)
print(f"\nAkurasi: {akurasi * 100:.2f}%")
print("\nLaporan klasifikasi:")
laporan = classification_report(
    y_test,
    prediksi,
    labels=[0, 1, 2],
    target_names=["Normal", "Penipuan", "Promo"],
    zero_division=0
)

print(laporan)

matriks = confusion_matrix(
    y_test,
    prediksi,
    labels=[0, 1, 2]
)

tabel_matriks = pd.DataFrame(
    matriks,
    index=["Normal", "Penipuan", "Promo"],
    columns=["Normal", "Penipuan", "Promo"]
)

print("\nConfusion matrix:")
print("Baris = label asli, kolom = hasil prediksi")
print(tabel_matriks)



joblib.dump(model, folder / "model_sms.joblib")

folder_hasil = folder / "hasil"
folder_hasil.mkdir(exist_ok=True)

teks_evaluasi = (
    f"Akurasi: {akurasi * 100:.2f}%\n\n"
    + laporan
)
(folder_hasil / "evaluasi.txt").write_text(
    teks_evaluasi,
    encoding="utf-8"
)
tabel_matriks.to_csv(
    folder_hasil / "confusion_matrix.csv"
)

hasil_uji = pd.DataFrame({
    "teks": x_test,
    "label_asli": y_test,
    "prediksi": prediksi
})

hasil_uji.to_csv(
    folder_hasil / "prediksi_uji.csv",
    index=False
)

print("\nModel dan hasil evaluasi berhasil disimpan.")
