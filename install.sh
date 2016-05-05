#!/bin/bash
sudo apt-get update

sudo apt-get install postgresql-9.3
sudo su postgres
createdb immagini
psql immagini < dump_immagini.sql

sudo apt-get install libpq-dev
wget http://pqxx.org/download/software/libpqxx/libpqxx-4.0.tar.gz
tar xvfz libpqxx-4.0.tar.gz
cd libpqxx-4.0
chmod +x configure
./configure
make
sudo make install
rm -R libpqxx-4.0
rm -R libpqxx-4.0.tar.gz

sudo apt-get install cimg-dev
