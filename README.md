# HandPostureTranslator
### Manuel D'utilisation
#### Dépdenances 
Pour utiliser tous les scripts python présente dans le projet, il faut tout d'abord installer queleuqe dépendances qui sont : 
*  Tout d'abord l'interpréteur de [Python](https://python.org), de préférence la version 3.6.
*  De préférence l'IDE [PyCharm](https://www.jetbrains.com/pycharm/) pour faciliter la gestion des packages et le lancement des scripts.
*  Le gestionnaire de package de python `pip` en lançant la commande suivante dans un terminal linux : 
`sudo apt-get install pip` ou bien sous winwods suivre ce [tutoriel](https://github.com/BurntSushi/nfldb/wiki/Python-&-pip-Windows-installation) extrêmement long mais facile a comprendre.
*  L'ensemble des packages python utilisé pour ce projet, la list complète est disponible dans le fichier `Projet/python-packages-dependencies.txt`, il est possible des les installer en même temps en lançant la commande : `pip3 install -r Chemin/Vers/python-packages-dependencies.txt`


#### Installation des deux applications 
Pour utliser l'application mobile, il faut d'abord les installer en suivant ces deux étapes : 

* D'abord installer l'application desktop ( le serveur ) comme suit :

	1. Ouvrir le projet l'application desktop d'une des deux manières suivantes : 
		* Compiler le projet Java qui se trouve dans le répertoire : <br/>`Project/Annexe/Application/Application Desktop` en utilisant un IDE dédié.
		* Copier le `.jar` fournit dans : <br/>`Project/Annexe/Application/Application-Desktop/out/artifacts/GloveSimulator_jar/GloveSimulator.jar` dans un répertoire désiré

* Installer l'application sur sons téléphone :
	1. copier l'application au format `.apk` se trouvant dans  `Project/Annexe/Application/AppMobile.apk` sur son téléphone android.
	

#### Utilisation des applications.

1. S'assurer que l'oridnateur est connecté un réseau sans fil `wlan1`
2. Lancer l'application d'une des manière vues précédemment.
La fenêtre suivante devrait apparaître : <br/>	
	<p align="center">
  	<img src="Annexe/Application/Screenshots%20Application/Desktop%20simulation%20de%20gant.png"/>
	</p>
	Avec la liste des utilsateur connecté vide car aucun appareil mobile n'est encore connecté.


2. Se conntecter au réseau sans fil `wlan1` mentionné plus tôt.
	3. Lancer l'insallation et ouvrir l'application.
	<p align="center">
  	<img src="Annexe/Application/Screenshots%20Application/Accueil.png" width="35%"/>
	</p>
	3. Il s'agit ensuite de s'identifier avec un username puis d'appuyer sur Look-For-Glove 
	
	<p align="center">
  	<img src="Annexe/Application/Screenshots Application/Recherche gant.png" width="35%"/>
	</p>
	
	Le serveur est listé on se connecte en appuyant dessus.



