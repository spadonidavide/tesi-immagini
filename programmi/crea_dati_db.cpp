
#ifndef CREA_DATI_CPP
#define CREA_DATI_CPP
	 
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

	
using namespace std;

string sql;




immagine load_file(const char *file_name, string path){

	immagine img(path);
	string buffer;
	float peso;
	ifstream f(file_name);
	punto p1,p2,p3,p4;
	if(f.is_open()) {
		while(getline(f, buffer)) {
			peso = atof(buffer.c_str());
			
			getline(f, buffer);
			p1.r = atoi(buffer.c_str());
			getline(f, buffer);
			p1.g = atoi(buffer.c_str());
			getline(f, buffer);
			p1.b = atoi(buffer.c_str());
			
			getline(f, buffer);
			p2.r = atoi(buffer.c_str());
			getline(f, buffer);
			p2.g = atoi(buffer.c_str());
			getline(f, buffer);
			p2.b = atoi(buffer.c_str());
			
			getline(f, buffer);
			p3.r = atoi(buffer.c_str());
			getline(f, buffer);
			p3.g = atoi(buffer.c_str());
			getline(f, buffer);
			p3.b = atoi(buffer.c_str());
			
			getline(f, buffer);
			p4.r = atoi(buffer.c_str());
			getline(f, buffer);
			p4.g = atoi(buffer.c_str());
			getline(f, buffer);
			p4.b = atoi(buffer.c_str());
			
			linea_img l(p1,p2,p3,p4,peso);
			img.add_linea(l);	
		}
		
		f.close();
	}
	else{
		throw "sticazzi\n";
	}
	
	return img;
}


/*
int aggiungi_immagine() {
	string path;
	immagine img;
	DB_manager manager;
	try {
		while(true) {
			cout<<"inserisci il path dell'immagine"<<endl;
			cin>>path;
			
			if(path=="esci")
				break;
			
			img = load_file("bezier.txt", path);		
			manager.save_image(img);
			

		}
	}catch (const std::exception &e){
			cerr << e.what() << std::endl;
	}
	
	return 0;
}

*/


/*
	try{
		connection C("dbname=postgres user=postgres password=segreto hostaddr=127.0.0.1 port=5432");
		if (C.is_open()) {
			cout << "Opened database successfully: " << C.dbname() << endl;
		} 
		else {
			cout << "Can't open database" << endl;
	 		return 1;
		}
		
		
		srand(time(NULL));
		for(int i=0; i<N_IMMAGINI; ++i){
			work W(C);
			string nome = "nome" + to_string(i);
			sql = "INSERT INTO Immagini(nome) VALUES('"+nome+"');";
			W.exec( sql );	
			W.commit();
			
			sql = "select max(id_immagini) from immagini";
			nontransaction N(C);
			result R( N.exec( sql ));
			result::const_iterator c = R.begin(); 
			int id_immagine = c[0].as<int>();
			N.commit();
			
			
			for(int j=0; j<L_IMMAGINI; ++j) {
				int id_punti[4];
				for(int k=0; k<4; ++k){
					int r,g,b;
	
					r = rand() % 16;
					g = rand() % 16;
					b = rand() % 16;
					
					sql = "select id_punti from punti where ";
					sql += to_string(r) +"=r and g="+to_string(g) +" and b="+to_string(b) +";";
					
					nontransaction N(C);
					result R( N.exec( sql ));
					N.commit();
					
					if(R.size()>0) {
						result::const_iterator c = R.begin();
						id_punti[k] = c[0].as<int>();
					}
					else {
						work W1(C);
						sql = "INSERT INTO punti(r,g,b) VALUES(";
						sql += to_string(r) +","+to_string(g) +","+to_string(b) +");";
								
						W1.exec(sql);
						W1.commit();
								
						sql = "select max(id_punti) from punti";
						nontransaction N(C);
						result R( N.exec( sql ));
						result::const_iterator c = R.begin(); 
						id_punti[k] = c[0].as<int>();
						N.commit();
					}
				}
				

				work W1(C);
				sql = "INSERT INTO linee(l1, l2, l3, l4) VALUES(";
				sql += to_string(id_punti[0]) +","+to_string(id_punti[1]);
				sql += ","+to_string(id_punti[2]) +","+to_string(id_punti[3]) +");";
					
				cout << sql << endl;
								
				W1.exec(sql);
				W1.commit();
					
				sql = "select max(id_linee) from linee;";
				nontransaction N1(C);
				result R1( N1.exec( sql ));
				result::const_iterator c = R1.begin(); 
				int id_linea = c[0].as<int>();
				N1.commit();
					
				work W2(C);
				sql = "INSERT INTO possiede(fk_id_linee, fk_id_immagini) VALUES(";
				sql += to_string(id_linea) +","+to_string(id_immagine)+");";
					
				W2.exec(sql);
				W2.commit();
			}		
			
		}
		
		C.disconnect ();
   }catch (const std::exception &e){
      cerr << e.what() << std::endl;
      return 1;
   }


		
		int id_immagini[N_IMMAGINI];
		sql = "SELECT id_immagini FROM immagini";
		nontransaction N(C);
		result R( N.exec( sql ));
		int n_id_im=0;
		for (result::const_iterator c = R.begin(); c != R.end(); ++c) {
			id_immagini[n_id_im] = c[0].as<int>();
			++n_id_im;
		}
		
		N.commit();
		work W1(C);
		srand(time(NULL));
		//genero i punti casualmente
		for(int i=0; i<N_IMMAGINI; ++i){ 
			for(int j=0; j<L_IMMAGINI; ++j) {
				for(int k=0; k<4; ++k){
					int r,g,b,rp,gp,bp;
	
					r = rand() % 16;
					g = rand() % 16;
					b = rand() % 16;
					rp = rand() % 16;
					gp = rand() % 16;
					bp = rand() % 16;
					
					sql = "select id_punti from punti where ";
					sql += to_string(r) +","+to_string(g) +","+to_string(b) +","+to_string(rp) +","
						+to_string(gp) +","+to_string(bp) +");";

					sql = "INSERT INTO punti(r,g,b,rp,gp,bp) VALUES(";
					sql += to_string(r) +","+to_string(g) +","+to_string(b) +","+to_string(rp) +","
						+to_string(gp) +","+to_string(bp) +");";
					
					cout<<sql<<endl;
					W1.exec( sql );
				}
				
			}
		}

		W1.commit();
		
		int id_linee[N_IMMAGINI*L_IMMAGINI];
		sql = "SELECT id_punti FROM punti";
		nontransaction N1(C);
		result R1( N1.exec( sql ));
		int n_id=0;
		N1.commit();
		
		for (result::const_iterator c = R1.begin(); c != R1.end(); ++c) {
			id_linee[n_id] = c[0].as<int>();
			++n_id;
		}
		
		work W2(C);
		for(int i=0; i<N_IMMAGINI; ++i) {
			for(int j=0; j<L_IMMAGINI;++j){
				
				sql = "INSERT INTO Possiede(fk_id_immagini, fk_id_linee) VALUES(";
				sql += to_string(id_immagini[i]) + "," + to_string(id_linee[j+i*L_IMMAGINI]) + ");";
				W2.exec( sql );
			}	
			
		}
		W2.commit();
		

		cout << "Records created successfully" << endl;
				
		*/
		
		
		
#endif

