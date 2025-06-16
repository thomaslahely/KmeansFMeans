import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re

#Lire le contenu du fichier tweets.txt et l'affecter à une variable texte.
with open ('tweets.txt', 'r') as lignes:
    texte = lignes.read()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def cleanWorld(texte):

    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F" 
                           u"\U0001F300-\U0001F5FF" 
                           u"\U0001F680-\U0001F6FF" 
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           "]+", flags=re.UNICODE)
    texte= emoji_pattern.sub(r'', texte)
    #Enlever les ponctuations.
    texte = texte.translate(str.maketrans('', '', string.punctuation))

    #Découper les mots en utilisant une méthode de tokenization
    tokens = word_tokenize(texte)

    #Mettre les mots en minuscule
    tokens_miniscule = [texte.lower() for texte in tokens]

    #Éliminer les stop words
    stop_words=set(stopwords.words('english'))
    tokend_sans_stopwords = [ texte for texte in tokens_miniscule if texte not in stop_words]

    #Appliquer le stemming sur les mots
    ps = PorterStemmer()
    tokens_stemming=[ps.stem(texte) for texte in tokend_sans_stopwords]

    #Eliminer les emojis


    #Retourner la liste des mots résultants sous forme d’une chaîne.
    return ' '.join(tokens_stemming)

#Appeler la fonction cleanWorld avec le contenu du fichier tweets.txt
res = cleanWorld(texte)

#Et on affiche le résultat
print(res)