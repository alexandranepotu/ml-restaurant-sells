# Machine Learning – Restaurant Sells (2025)

## Descriere Generală
Proiect de machine learning pentru analiza datelor de tranzacții dintr-un restaurant, cu scopul de a prezice comportamentul clienților și de a oferi recomandări de produse. Implementează algoritmi de clasificare de la zero și construiește sisteme de ranking pentru upselling.

## Configurare
Proiectul este implementat în Python, folosind un mediu virtual.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Pachete utilizate: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, tqdm.

## Dataset

Datasetul conține tranzacții realizate într-un restaurant:
- Fiecare rând reprezintă un produs de pe un bon fiscal
- Bonurile sunt identificate prin `id_bon`

### Statistici:
- 28.039 linii de tranzacții
- 7.869 bonuri fiscale
- 59 produse unice
- Interval: 5 septembrie - 3 decembrie 2025

### Coloane principale:
- `id_bon` - ID bon (identifică un coș/o tranzacție)
- `data_bon` - Data/ora bonului
- `retail_product_name` - Numele produsului
- `SalePriceWithVAT` - Preț per linie (cu TVA)

## Structura Proiectului

```
ml-restaurant-sells/
├── data/
│   └── ap_dataset.csv          # Date brute de tranzacții
├── output/                      # Vizualizări și rezultate generate
├── src/
│   ├── load_data.py            # Încărcare și explorare date
│   ├── build_baskets.py        # Reconstrucția coșurilor
│   ├── build_features.py       # Feature engineering
│   └── lr_crazy_sauce.py       # LR #1: Predicția sosurilor
├── requirements.txt
└── README.md
```

## Pipeline de Procesare a Datelor

### 1. Încărcare și Explorare Date
Verifică structura și afișează statistici de bază.

**Script:** `src/load_data.py`

```powershell
python src/load_data.py
```

### 2. Reconstrucția Coșurilor
Agregarea datelor la nivel de bon fiscal și extragerea trăsăturilor:
- `cart_size` - Număr de produse în coș
- `distinct_products` - Număr de produse distincte
- `total_value` - Valoarea totală a coșului
- `day_of_week`, `hour`, `is_weekend` - Trăsături temporale

**Script:** `src/build_baskets.py`

```powershell
python src/build_baskets.py
```

### 3. Feature Engineering
Construirea matricei de count encoding pentru produse și combinarea cu trăsăturile de coș și timp.

**Dataset final:** 7.869 bonuri × 66 trăsături (59 produse + 7 trăsături coș/timp)

**Script:** `src/build_features.py`

```powershell
python src/build_features.py
```

## Sarcini Machine Learning

### LR #1: Predicția Crazy Sauce (Implementat ✓)

**Problemă:** Considerând bonurile care conțin Crazy Schnitzel, preziceți dacă clientul va cumpăra și Crazy Sauce.

**Implementare:**
- Regresie Logistică custom cu Gradient Descent (implementat de la zero)
- Clasificare binară cu split 80/20 train-test
- Scalare trăsături cu StandardScaler
- Regularizare L2

**Script:** `src/lr_crazy_sauce.py`

```powershell
python src/lr_crazy_sauce.py
```

**Rezultate:**
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Custom LR | 96.08% | 97.31% | 95.26% | 96.28% | 0.9867 |
| Sklearn LR | 98.04% | 99.46% | 96.84% | 98.13% | 0.9991 |
| Baseline | 53.22% | 53.22% | 100% | 69.47% | 0.5000 |

**Observații cheie:**
- Coșuri mai mari (`cart_size`, `distinct_products`) prezic puternic cumpărarea sosului
- Garniturile (cartofi prăjiți, cartofi copți) cresc probabilitatea de achiziție a sosului
- Prezența altor sosuri (Cheddar, Garlic, Blueberry) scade probabilitatea pentru Crazy Sauce

**Rezultate generate:**
- `output/lr_crazy_sauce_results.png` - Curba de antrenare, comparație ROC, matrice de confuzie, analiză coeficienți


### LR #2: Multi-Sauce Recommendation (Implementat ✓)

**Problemă:** Pentru fiecare sos din listă, se antrenează un model de regresie logistică care prezice dacă sosul respectiv va fi cumpărat pe un bon (y_s = 1 dacă s apare în bon, altfel 0). Pentru un coș dat (fără sosuri), se calculează P(s | coș) pentru fiecare sos și se recomandă Top-K sosuri cu probabilitatea cea mai mare (care nu sunt deja în coș).

**Implementare:**
- Un model Logistic Regression (scikit-learn) pentru fiecare sos
- Trăsături: count encoding pentru produse (excluzând sosul curent), cart_size, distinct_products, total_value, day_of_week, is_weekend
- Împărțire train/test la nivel de bon
- Recomandare Top-K sosuri pentru fiecare coș
- Evaluare Hit@K: cât de des sosul real din bon apare în Top-K recomandări
- Comparare cu baseline de popularitate globală (cele mai populare sosuri)
- Vizualizare coeficienți (importanța trăsăturilor) pentru fiecare model de sos (salvate în `output/lr2_top_features_[nume_sos].png`)

**Script:** `src/lr_all_sauces.py`

```powershell
python src/lr_all_sauces.py
```

**Rezultate generate:**
- Hit@K (LR models): Procentul de bonuri pentru care cel puțin un sos real a fost recomandat în Top-K
- Hit@K (Popularitate): Procentul de bonuri pentru care cel puțin un sos real se află în Top-K cele mai populare sosuri
- `output/lr2_top_features_[nume_sos].png` - Top 15 coeficienți pentru fiecare model de sos

**Exemplu output:**
```
Hit@3 (LR models):     78.12%
Hit@3 (Popularity):    65.34%
```

**Observații cheie:**
- Modelele LR personalizează recomandarea pe conținutul coșului, depășind baseline-ul de popularitate
- Analiza coeficienților arată ce produse cresc/scad probabilitatea de a cumpăra fiecare sos




### Ranking pentru Upselling cu Naive Bayes (Implementat ✓)

**Cerința 3:** Pentru un coș parțial și un context fix, se produce o ierarhie (ranking) de produse candidate pentru “upsell”, în funcție de venitul și popularitatea aferente acestora.

**Implementare:**
- S-a implementat un sistem de ranking folosind Naive Bayes (BernoulliNB) pentru a estima probabilitatea P(p | coș) pentru fiecare produs candidat (ex: sosuri).
- Scorul de ranking folosit: `Score(p | coș) = P(p | coș) × price(p)` (valoare așteptată maximă).
- Pentru fiecare produs, se antrenează un model BernoulliNB (scikit-learn) pe coșuri, folosind count encoding binar pentru produse și trăsături de coș.
- Pentru un coș dat, se exclud produsele deja prezente și se recomandă Top-K produse cu scorul cel mai mare.
- Evaluare Hit@K: cât de des produsul eliminat din coș este recuperat de algoritm în Top-K recomandări.
- Comparare cu baseline de popularitate și logistic regression.

**Script:** `src/nb_ranking.py`

```powershell
python src/nb_ranking.py
```

**Rezultate generate:**
- Hit@K (Naive Bayes): Procentul de bonuri pentru care produsul real eliminat a fost recomandat în Top-K
- Exemplu de recomandare pentru un coș parțial

**Exemplu output:**
```
Hit@3 (Naive Bayes Ranking): 84.57%
Hit@3 (Popularity):         71.08%

Example NB recommendation for basket 12345:
['Crazy Sauce', 'Cheddar Sauce', 'Garlic Sauce']
```

**Observații cheie:**
- Naive Bayes permite ranking rapid și scalabil pentru orice subset de produse candidate
- Integrarea prețului în scor maximizează valoarea așteptată a recomandărilor

## Rularea Pipeline-ului Complet

```powershell
# 1. Explorare date
python src/load_data.py

# 2. Construirea coșurilor
python src/build_baskets.py

# 3. Construirea trăsăturilor
python src/build_features.py

# 4. Rulare LR #1
python src/lr_crazy_sauce.py
```

## Note
- Dataset-ul este pentru uz didactic
- Nu conține date personale despre clienți
- Toate scripturile urmează structură și convenții de denumire consistente
