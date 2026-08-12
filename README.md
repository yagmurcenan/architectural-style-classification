# 🏛️ Architectural Style Classification

Mimari görselleri analiz ederek **36 farklı mimari stilden** hangisine ait olduğunu tahmin eden, web ve mobil cihazlarla uyumlu bir **derin öğrenme uygulamasıdır**.

Uygulamada **EfficientNetB0** tabanlı görüntü sınıflandırma modeli kullanılmıştır. Kullanıcı bir mimari görsel yüklediğinde model, tahmin edilen stili ve güven skorunu belirler. Ayrıca ilgili mimari stilin öncüsü ve stil hakkında açıklayıcı bilgiler sunulur.

## ✨ Features

- 🏛️ **36 farklı mimari stilin** sınıflandırılması
- 🧠 **EfficientNetB0** tabanlı görüntü sınıflandırma
- 🖼️ Görsel yükleyerek otomatik tahmin
- 🎯 Tahmin güven skorunun gösterilmesi
- 👤 Stil öncüsü ve stil açıklamasının sunulması
- 📱 Web ve mobil uyumlu arayüz
- 📊 F1-Score ve Cohen's Kappa ile model değerlendirmesi

## 🧠 Model

Model, **EfficientNetB0** mimarisi kullanılarak 36 farklı mimari stil üzerinde eğitilmiş ve test edilmiştir.

Tahmin sonucunda kullanıcıya:

- 🏛️ Mimari stil
- 🎯 Güven skoru
- 👤 Stil öncüsü
- 📖 Stil açıklaması

sunulmaktadır.

## 🔄 Application Workflow

```text
Architectural Image
        │
        ▼
Image Preprocessing
        │
        ▼
EfficientNetB0
        │
        ▼
Style Prediction
        │
        ├── Style
        ├── Confidence Score
        ├── Style Pioneer
        └── Description
        │
        ▼
Web / Mobile Interface

## 📊 Model Performance

| Metric | Score |
|---|---:|
| **F1-Score** | ~90% |
| **Cohen's Kappa** | ~90% |

## 🛠️ Technologies

- Python
- TensorFlow / Keras
- EfficientNetB0
- Deep Learning
- Computer Vision
- Image Classification
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
