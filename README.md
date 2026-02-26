# Klasifikacija ptica Ulcinjske Solane

Klasifikacija 10 vrsta najčešće viđenih ptica karakterističnih za područje Ulcinjske Solane primenom 
Transfer Learning pristupa zasnovanog na modelu EfficientNet-B3, prethodno 
treniranom na ImageNet skupu podataka.

## Dataset
Zbog velikog broja slika potrebnih za treniranje (oko 1000 ukupno, 100 po vrsti), slike se ne nalaze u sklopu repozitorijuma. Fotografije su prikupljene sa:
- [iNaturalist](https://www.inaturalist.org/places/ulcinj) — opservacije ptica sa područja Ulcinja
- [CUB-200-2011](https://www.kaggle.com/datasets/wenewone/cub2002011) — javno dostupan ornitološki skup podataka

Folder `data/` treba da sadrži po jedan podfolder za svaku vrstu ptice sa 
odgovarajućim fotografijama. Test skup je već izdvojen u folderu `test_data/`.

## Instalacija
```bash
pip install -r requirements.txt
```

## Pokretanje

Trening i evaluacija:
```bash
python main.py
```

Samo evaluacija (potreban sačuvan model u `models/`):
```bash
python evaluate.py
```

## Rezultati
Model postiže tačnost od 93% na test skupu. Rezultati evaluacije 
(confusion matrix i krive učenja) čuvaju se u folderu `results/`.
