# Machine Learning – Restaurant Sells (2025)

## Setup
Proiectul este implementat în Python, folosind un mediu virtual.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt


Pachete utilizate: pandas, numpy, scikit-learn, matplotlib (seaborn, scipy, tqdm – opțional).

## Dataset

Datasetul conține tranzacții realizate într-un restaurant:

fiecare rând reprezintă un produs de pe un bon fiscal;

bonurile sunt identificate prin id_bon.

## Statistici:

28.039 linii;

7.869 bonuri fiscale;

59 produse unice;

interval: 05.09.2025 – 03.12.2025.

Coloane utilizate:

id_bon

data_bon

retail_product_name

SalePriceWithVAT

## Data processing

Datele au fost procesate în trei etape:

Încărcare și explorare
Verificarea structurii și statisticilor de bază.

Script: src/load_data.py

Reconstrucția coșurilor
Agregarea datelor la nivel de bon fiscal și extragerea trăsăturilor:

cart_size

distinct_products

total_value

day_of_week, hour, is_weekend

Script: src/build_baskets.py

Feature engineering
Construirea vectorilor de produse (count encoding) și combinarea acestora cu trăsăturile de coș și timp.

Dataset final: 7.869 bonuri × 66 trăsături

Script: src/build_features.py