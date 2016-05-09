#!/bin/bash
rm path_immagini_ind.txt;
mogrify -resize 128x128! /home/$USER/Immagini/immagini/*.jpg
for f in /home/$USER/Immagini/immagini/*.jpg
do
	echo $f >> path_immagini_ind.txt;
done

./a.out;
for f in /home/$USER/Immagini/immagini/*.jpg
do
	mv $f /home/$USER/Immagini/immagini/immagini_caricate;
done
