import matplotlib.pyplot as plt
import pandas as pad
import seaborn as sbn
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score,classification_report

# on charge le jeu de données
donnee = pad.read_csv('US_Airlines_Twitter_Sentiment.csv')

#le nombre de tweets pour chaque compagnie aérienne.
plt.figure(figsize=(10, 6))
sbn.countplot(x='airline',data=donnee)
plt.title('Nombre de tweets par compagnie aérienne')
plt.show()

#le nombre de tweets pour chaque sentiment.
plt.figure(figsize=(10, 6))
sbn.countplot(x='airline_sentiment',data=donnee)
plt.title('Nombre de tweets par sentiment')
plt.show()

#le nombre de tweets pour chaque sentiment par compagnie aérienne.
plt.figure(figsize=(10, 6))
sbn.countplot(x='airline',hue='airline_sentiment',data=donnee)
plt.title('sentiment par compagnie aérienne')
plt.show()

def NettoyerTexte(texte):
    texte = re.sub(r'http\S+', '', texte)
    texte = re.sub(r'@\w+', '', texte)
    texte = re.sub(r'#\w+', '', texte)
    texte = re.sub(r'[^\w\s]', '', texte)

    return texte
donnee['texte'] = donnee['text'].apply(NettoyerTexte)


vectorizer = TfidfVectorizer(stop_words='english')
x = vectorizer.fit_transform(donnee['texte'])
y=donnee['airline_sentiment']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# on crée un classifieur RandomForest
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(x_train, y_train)

# on utliser le modèle pour prédire les sentiments des tweets de l'ensemble de test
y_pred = classifier.predict(x_test)
# on affiche le rapport de classification
print(classification_report(y_test, y_pred))

# on affiche la matrice de confusion
confusion_matrice = confusion_matrix(y_test, y_pred)
sbn.heatmap(confusion_matrice, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de confusion')
plt.xlabel('Prédictions')
plt.ylabel('Vérités terrain')
plt.show()

# pourcentage de prédiction correcte sur l'ensemble de test.
print('précision:', accuracy_score(y_test, y_pred))

#F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print('F1 score:', f1)