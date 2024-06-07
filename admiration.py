from transformers import pipeline

pipe = pipeline("text-classification", model="SchuylerH/bert-multilingual-go-emtions")

texte = "you're so nice"
resultats = pipe(texte)

for resultat in resultats:
    print(f"Label: {resultat['label']}, Score: {resultat['score']:.4f}")
