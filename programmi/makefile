ricerca_db: ricerca_db.cpp
	g++ ricerca_db.cpp -o ricerca_db -lpqxx -lpq -O2 -L/usr/X11R6/lib -lm -lpthread -lX11

crea_dati: crea_dati_db.cpp
	g++ crea_dati_db.cpp -o crea_dati -lpqxx -lpq -W -Wall -ansi -pedantic -Dcimg_use_vt100 -Dcimg_display=0 -fno-tree-pre -lm

DB_manager.o: DB_manager.cpp
	g++ DB_manager.cpp -c DB_manager.o

immagini.o: immagini.cpp
	g++ immagini.cpp -c immagini.o
	
linee.o: linee.cpp
	g++ linee.cpp -c linee.o

crea_dati.o: crea_dati_db.cpp 
	g++ crea_dati_db.cpp -c crea_dati.o
	
crea_dati___: crea_dati.o immagini.o DB_manager.o linee.o
	g++ crea_dati.o immagini.o DB_manager.o linee.o -o crea_immagini -lpqxx -lpq
	
ricerca: ricerca.cpp I_manager.cpp DB_manager.cpp
	g++ ricerca.cpp -o ricerca -w -Dcimg_use_vt100 -fno-tree-pre -lm -O2 -L/usr/X11R6/lib -lm -lpthread -lX11 -std=c++11 -lpqxx -lpq
	

