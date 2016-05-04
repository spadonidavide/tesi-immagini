test: test.c indicizza_immagini_opt.cpp
	g++ indicizza_immagini_opt.cpp -w -Dcimg_use_vt100 -Dcimg_display=0 -fno-tree-pre -lm -lpqxx -lpq -lpthread
	
crea_dati: crea_dati_db.cpp
	g++ crea_dati_db.cpp -o crea_dati -Wall -W -ansi -pedantic -lpqxx -lpq  -Dcimg_use_vt100 -Dcimg_display=0 -fno-tree-pre -lm
	
	
prova: prova.cpp
	g++ prova.cpp -o prova -w -Dcimg_use_vt100 -Dcimg_display=0 -fno-tree-pre -lm -lpqxx -lpq -lpthread
