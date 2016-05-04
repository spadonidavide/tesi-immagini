#!/bin/bash
rm path_immagini_ind.txt;
for f in immagini/*.jpg
do
	echo $f >> path_immagini_ind.txt;
done

./a.out;
for f in immagini/*.jpg
do
	mv $f /home/davide/Immagini/immagini/immagini_caricate;
done
