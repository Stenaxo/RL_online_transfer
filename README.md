## RL_online_transfer

#Environnement python

Un fichier requirements.txt est a disposition pour installer toutes les dépendances nécessaires.

```
pip install -r requirements.txt
```

#Téléchargement des données

Il faut télécharger la base de données de la manière suivante tout en la placant dans le même répertoire que le code. Modifier le cd avec votre répertoire.

```
cd ./data
wget http://csr.bu.edu/ftp/visda17/clf/train.tar
tar xvf train.tar

wget http://csr.bu.edu/ftp/visda17/clf/validation.tar
tar xvf validation.tar  

wget http://csr.bu.edu/ftp/visda17/clf/test.tar
tar xvf test.tar

wget https://raw.githubusercontent.com/VisionLearningGroup/taskcv-2017-public/master/classification/data/image_list.txt
```

Pour plus d'informations concernant les données ou leur utilisation : https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification