#ifndef DB_MANAGER_CPP
#define DB_MANAGER_CPP

#include <string>
#include <iostream>
#include <pqxx/pqxx>
#include "immagine.cpp"
#include "match_prof.cpp"
#include <algorithm>

using namespace std;
using namespace pqxx;



class DB_manager {

public:
	//crea una nuova immagine e torna l'id appena generato
	/*
	int new_image(string url){
		int id_immagine;
		try{
			connection C("dbname=postgres user=postgres password=segreto hostaddr=127.0.0.1 port=5432");
			if (C.is_open()) {
				//cout << "Opened database successfully: " << C.dbname() << endl;
			} 
			else {
				cout << "Can't open database" << endl;
			}


			work W(C);
			string sql = "INSERT INTO Immagini(nome) VALUES('"+url+"');";
			W.exec( sql );	
			W.commit();
	
			sql = "select max(id_immagini) from immagini";
			nontransaction N(C);
			result R( N.exec( sql ));
			result::const_iterator c = R.begin(); 
			id_immagine = c[0].as<int>();
			N.commit();
	
			C.disconnect ();
		
		}catch (const std::exception &e){
			cerr << e.what() << std::endl;
		}
	
	
		return id_immagine;
	}

	int find_point(int r, int g, int b) {
		int id_punto;
		try{
			connection C("dbname=postgres user=postgres password=segreto hostaddr=127.0.0.1 port=5432");
			if (C.is_open()) {
				//cout << "Opened database successfully: " << C.dbname() << endl;
			} 
			else {
				cout << "Can't open database" << endl;
			}

			string sql = "select id_punti from punti where ";
			sql += to_string(r) +"=r and g="+to_string(g) +" and b="+to_string(b) +";";
			
			nontransaction N(C);
			result R( N.exec( sql ));
			N.commit();
		
			
			if(R.size()>0) {
				result::const_iterator c = R.begin();
				id_punto = c[0].as<int>();
			}
			else
				id_punto = -1;
		
			C.disconnect ();
				
		}catch (const std::exception &e){
			cerr << e.what() << std::endl;
		}
			
		return id_punto;
			
	}
		
	int new_point(int r, int g, int b) {
		int id_punto;
		try{
			connection C("dbname=postgres user=postgres password=segreto hostaddr=127.0.0.1 port=5432");
			if (C.is_open()) {
				//cout << "Opened database successfully: " << C.dbname() << endl;
			} 
			else {
				cout << "Can't open database" << endl;
			}
		
			work W(C);
			string sql = "INSERT INTO punti(r,g,b) VALUES(";
			sql += to_string(r) +","+to_string(g) +","+to_string(b) +") RETURNING id_punti;";
						
			result R( W.exec( sql ));
			result::const_iterator c = R.begin(); 
			id_punto = c[0].as<int>();
			
			W.commit();
		
			C.disconnect();
		}catch (const std::exception &e){
			cerr << e.what() << std::endl;
		}
	
		return id_punto;
	}
		
	int new_line(int *id_punti, float peso) {
		int id_linea;
		try{
			connection C("dbname=postgres user=postgres password=segreto hostaddr=127.0.0.1 port=5432");
			if (C.is_open()) {
				//cout << "Opened database successfully: " << C.dbname() << endl;
			} 
			else {
				cout << "Can't open database" << endl;
			}
		
			work W(C);
			string sql = "INSERT INTO linee(l1, l2, l3, l4,peso) VALUES(";
			sql += to_string(id_punti[0]) +","+to_string(id_punti[1]);
			sql += ","+to_string(id_punti[2]) +","+to_string(id_punti[3])+","+to_string(peso)+") RETURNING id_linee;";
			
					
			result R( W.exec( sql ));
			result::const_iterator c = R.begin(); 
			id_linea = c[0].as<int>();
			
			W.commit();	
				
			work W2(C);
			sql = "INSERT INTO possiede(fk_id_linee, fk_id_immagini) VALUES(";
			sql += to_string(id_linea) +","+to_string(id_immagine)+");";
			
			W2.exec(sql);
			W2.commit();
			
		
			C.disconnect();
		}catch (const std::exception &e){
			cerr << e.what() << std::endl;
		}
	
		return id_linea;
	}	
	
		
	void save_possiede(int id_immagine, int *id_linee, int n_linee){

		try{
			connection C("dbname=postgres user=postgres password=segreto hostaddr=127.0.0.1 port=5432");
			if (C.is_open()) {
				//cout << "Opened database successfully: " << C.dbname() << endl;
			} 
			else {
				cout << "Can't open database" << endl;
			}
		
			work W(C);
			string sql;
			for(int i=0; i<n_linee; ++i) {
				sql = "INSERT INTO possiede(fk_id_linee, fk_id_immagini) VALUES(";
				sql += to_string(id_linee[i]) +","+to_string(id_immagine)+");";
			
				W.exec(sql);
			}
			W.commit();
		
			C.disconnect();
		}catch (const std::exception &e){
			cerr << e.what() << std::endl;
		}	
	}*/


	void load_db(result::const_iterator &begin, result::const_iterator &end) {
		
		try{
			connection C("dbname=postgres user=postgres password=segreto hostaddr=127.0.0.1 port=5432");
			if (C.is_open()) {
				//cout << "Opened database successfully: " << C.dbname() << endl;
			} 
			else {
				cout << "Can't open database" << endl;
			}
		
			string sql = "SELECT id_immagini,id_linee,r,g,b FROM linee,immagini,possiede,punti ";
			sql += "WHERE id_immagini=fk_id_immagini AND id_linee=fk_id_linee and ";
			sql += "(id_punti=l1 OR id_punti=l2 OR id_punti=l3 OR id_punti=l4) ";
			sql += "ORDER BY id_immagini, id_linee, id_punti;";
			nontransaction N(C);
			result R( N.exec( sql ));


			cout<<"num di record caricati: "<<R.size()<<endl;

				
			//carico tutto il db
			begin = R.begin();
			end = R.end();
			
			/*
			punto p;
			int id_linea, id_immagine, id_linea_vecchio;
			while (c != R.end()) {
				id_immagine = c[0].as<int>();
				immagine img;
				img.set_id_immagine(id_immagine);
			
				id_linea = c[1].as<int>();
				id_linea_vecchio = id_linea;
				while(id_linea==id_linea_vecchio) {
					for(int i=0; i<4; ++i){
						p[i].r = c[2].as<int>();
						p[i].g = c[3].as<int>();
						p[i].b = c[4].as<int>();
						++c;
					}
			
					linea l(p[0], p[1], p[2], p[3]);
					l.set_id_linea(id_linea);
					img.add_linea(l);
					++c;
					id_linea = c[1].as<int>();
				}
			
				immagini.push_back(img);
			}*/
		
			C.disconnect();
		}catch (const std::exception &e){
			cerr << e.what() << std::endl;
		}
		
		//return immagini;
	}
	
	void save_image(immagine img) {
		try{
			connection C("dbname=postgres user=postgres password=segreto hostaddr=127.0.0.1 port=5432");
			if (C.is_open()) {
				//cout << "Opened database successfully: " << C.dbname() << endl;
			} 
			else {
				cout << "Can't open database" << endl;
			}


			work W(C);
			string sql = "INSERT INTO Immagini(nome) VALUES('"+img.get_url()+"') RETURNING id_immagini;";
			
			result R( W.exec( sql ));
			result::const_iterator c = R.begin(); 
			int id_immagine = c[0].as<int>();
			
			vector<linea_img> linee = img.get_linee();
			int id_linee[linee.size()];
			
			for(unsigned i=0; i<linee.size(); ++i) {
				linea_img l = linee[i];
				int id_punti[4];
				
				//salvo i punti
				sql = "INSERT INTO punti(r,g,b) VALUES(";
				sql += to_string(l.get_p1().r) +","+to_string(l.get_p1().g) +","+to_string(l.get_p1().b);
				sql += ") RETURNING id_punti;";						
				result R1( W.exec( sql ));
				result::const_iterator c = R1.begin(); 
				id_punti[0] = c[0].as<int>();
				
				
				sql = "INSERT INTO punti(r,g,b) VALUES(";
				sql += to_string(l.get_p2().r) +","+to_string(l.get_p2().g) +","+to_string(l.get_p2().b);
				sql += ") RETURNING id_punti;";									
				result R2( W.exec( sql ));
				c = R2.begin(); 
				id_punti[1] = c[0].as<int>();
				
				
				sql = "INSERT INTO punti(r,g,b) VALUES(";
				sql += to_string(l.get_p3().r) +","+to_string(l.get_p3().g) +","+to_string(l.get_p3().b);
				sql += ") RETURNING id_punti;";
				result R3( W.exec( sql ));
				c = R3.begin(); 
				id_punti[2] = c[0].as<int>();
				
				sql = "INSERT INTO punti(r,g,b) VALUES(";
				sql += to_string(l.get_p4().r) +","+to_string(l.get_p4().g) +","+to_string(l.get_p4().b);
				sql += ") RETURNING id_punti;";
				result R4( W.exec( sql ));
				c = R4.begin(); 
				id_punti[3] = c[0].as<int>();
			
				
				//salvo la linea
				string sql = "INSERT INTO linee(l1, l2, l3, l4,peso) VALUES(";
				sql += to_string(id_punti[0]) +","+to_string(id_punti[1]);
				sql += ","+to_string(id_punti[2]) +","+to_string(id_punti[3])+","+to_string(l.get_peso());
				sql += ") RETURNING id_linee;";
			
					
				result R5( W.exec( sql ));
				c = R5.begin(); 
				id_linee[i] = c[0].as<int>();
			
				
			}

			//salvo la tabella possiede
			for(unsigned i=0; i<linee.size(); ++i) {
				sql = "INSERT INTO possiede(fk_id_linee, fk_id_immagini) VALUES(";
				sql += to_string(id_linee[i]) +","+to_string(id_immagine)+");";
			
				W.exec(sql);
			}
			
			W.commit();
			C.disconnect ();
		
		}catch (const std::exception &e){
			cerr << e.what() << std::endl;
		}
	}
	
	
	bool exist_image(string id_img) {
		try{
			connection C("dbname=postgres user=postgres password=segreto hostaddr=127.0.0.1 port=5432");
			if (C.is_open()) {
				//cout << "Opened database successfully: " << C.dbname() << endl;
			} 
			else {
				cout << "Can't open database" << endl;
			}
	
	
			string sql = "select id_immagini from immagini where nome='"+id_img+"';";
			nontransaction N(C);
			result R( N.exec( sql ));
			N.commit();
			
			C.disconnect();
		
			if(R.size()>0)
				return true;
			
			return false;

		}catch (const std::exception &e){
			cerr << e.what() << std::endl;
			return false;
		}		
	}
	
	vector<string> ordina_classifica(vector<string> paths) {
		vector<float> scores;
		for(int i=0; i<paths.size(); ++i) {
			cout<<"immagine numero:"<<i<<endl;
			
			FILE* f = fopen("output2.txt", "w");
						
			string path = paths[i].substr(31, paths[i].length()-31);
			string sql = "SELECT peso, l1, l2, l3, l4 FROM linee, possiede, immagini WHERE nome='"+path+"'";
			sql += " and fk_id_immagini=id_immagini and fk_id_linee=id_linee;";
			
			try{
				connection C("dbname=postgres user=postgres password=segreto hostaddr=127.0.0.1 port=5432");
				if (C.is_open()) {
					//cout << "Opened database successfully: " << C.dbname() << endl;
				} 
				else {
					cout << "Can't open database" << endl;
				}
				work N(C);
				result R( N.exec( sql ));
				N.commit();
				
				for(result::const_iterator it = R.begin(); it!=R.end(); ++it) {
					float peso = it[0].as<float>();
					int id_p1 = it[1].as<int>();
					int id_p2 = it[2].as<int>();
					int id_p3 = it[3].as<int>();
					int id_p4 = it[4].as<int>();
					
					//prendo tutti gli rgb della linea
					sql = "SELECT r,g,b FROM punti WHERE id_punti="+to_string(id_p1)+";";
					work N1(C);
					result R1( N1.exec( sql ));
					N1.commit();
					
					sql = "SELECT r,g,b FROM punti WHERE id_punti="+to_string(id_p2)+";";
					work N2(C);
					result R2( N2.exec( sql ));
					N2.commit();
					
					sql = "SELECT r,g,b FROM punti WHERE id_punti="+to_string(id_p3)+";";
					work N3(C);
					result R3( N3.exec( sql ));
					N3.commit();
					
					sql = "SELECT r,g,b FROM punti WHERE id_punti="+to_string(id_p4)+";";
					work N4(C);
					result R4( N4.exec( sql ));
					N4.commit();
					
					int r[4];
					int g[4];
					int b[4];
					result::const_iterator ij = R1.begin();
					r[0] = ij[0].as<int>();
					g[0] = ij[1].as<int>();
					b[0] = ij[2].as<int>();
					
					ij = R2.begin();
					r[1] = ij[0].as<int>();
					g[1] = ij[1].as<int>();
					b[1] = ij[2].as<int>();
					
					ij = R3.begin();
					r[2] = ij[0].as<int>();
					g[2] = ij[1].as<int>();
					b[2] = ij[2].as<int>();
					
					ij = R4.begin();
					r[3] = ij[0].as<int>();
					g[3] = ij[1].as<int>();
					b[3] = ij[2].as<int>();
					
					//scrivo su file i dati
					fprintf(f, "%f ", peso);
					for(int i=0; i<3; ++i) {
						fprintf(f, "%d, %d, %d, ", r[i], g[i], b[i]);
					}
					fprintf(f, "%d, %d, %d\n", r[3], g[3], b[3]);				
				}
				
				C.disconnect();
			}catch (const std::exception &e){
				cerr << e.what() << std::endl;
			}
			fclose(f);
			scores.push_back(match_prof());
			//cout<<match_prof()<<endl;
		}
		
		int N_CLASSIFICA = 8;
		vector<int> scarti;
		vector<string> classifica;
		for(int i=0; i<N_CLASSIFICA; ++i) {
			float min = 1.0;
			int index_min = 0;
			for(int i=0; i<scores.size(); ++i) {
				if(scores[i]<min && find(scarti.begin(), scarti.end(), i)==scarti.end()) {
					index_min = i;
					min = scores[i];
				}
			}
			scarti.push_back(index_min);
			classifica.push_back(paths[index_min]);
		}
			
		//cout<<"immagine:"<<paths[i]<<" score:"<<scores[i]<<endl;
		return classifica;
	}
	
};	

#endif

