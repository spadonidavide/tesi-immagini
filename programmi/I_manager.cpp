#ifndef IMANAGER_CPP
#define IMANAGER_CPP

#include <fstream>
#include "immagine.cpp"
#include <stdlib.h>

struct id_iml{
	int id_immagine;
	int id_linea;
	float peso;
};

struct el_cube{
	float peso;
	int n_linea;
};

struct indice{
	int pindex;
	int id_img;
	int n_linea;
	float peso;
};

//path del file che contiene i dati indicizzati
static string indicizz_path = "/home/davide/Documenti/tesi/indicizzatore/indicizzatore.txt";

//path del file che contiene l-ultimo id inserito
static string last_id_path = "/home/davide/Documenti/tesi/indicizzatore/last_id.txt";

//path del file che contiene i path delle mmagini indicizzate
static string img_id_path = "/home/davide/Documenti/tesi/indicizzatore/path_immagini.txt";

class I_manager {
	private:
		
		
		el_cube cube[16][16][16];
		
		int get_pindex(int r, int g, int b) {
			int indice = r + g * 16 + b * pow(16, 2);
	
			return indice;
		}
		
		
		void reset_cube(){
			for(int x=0; x<16; ++x){
				for(int y=0; y<16; ++y){
					for(int z=0; z<16; ++z){
						cube[x][y][z].peso = 0;
						cube[x][y][z].n_linea = 0;
					}
				}
			}
		}
		
	public:
	
		//legge l'ultimo id inserito delle imagini
		int read_last_id() {
			fstream f(last_id_path.c_str());
			
			string buf;
			getline(f, buf);
						
			return atoi(buf.c_str());
		}

		//salva un nuovo id immagini
		void save_new_id(int id) {
			FILE* f=fopen(last_id_path.c_str(),"w");
			fprintf(f,"%d\n",id);
			fclose(f);
		}
		
		//aggiunge un nuovo path di un'immagine
		void add_path(string path) {
			FILE* f=fopen(img_id_path.c_str(),"a");
			fprintf(f,"%s\n",path.c_str());
			fclose(f);
		}
		
		
		//ritorna il path dell'iesima immagine
		string get_image_path(int pos) {
			ifstream f(img_id_path.c_str());
			string buf;
			for(int i=0; i<=pos; ++i) {
				getline(f,buf);
			}
			f.close();
			return buf;
		}
		
		//indicizza un'immagine nel file
		void indicizza(immagine img) {
			vector<linea_img> linee = img.get_linee();
			reset_cube();
			
			int n_linea = 1;
			//per ogni linea dell'immagine calcolo la bezier 
			for(vector<linea_img>::iterator it=linee.begin(); it<linee.end(); ++it) {
				linea_img l = *it;
				float peso = l.get_peso();
			
				list<punto> punti = bezier(l.get_p1(), l.get_p2(), l.get_p3(), l.get_p4());
				for(list<punto>::iterator ij=punti.begin(); ij!=punti.end(); ++ij) {
					punto p_app = *ij;
					int r = p_app.r;
					int g = p_app.g;
					int b = p_app.b;
			
					cube[r][g][b].n_linea = n_linea;
					
					cube[r][g][b].peso = peso;
				}
				++n_linea;
			}
			
			int id_immagine = read_last_id();
			++id_immagine;
			
			FILE* f=fopen(indicizz_path.c_str(),"a");
			//aggiungo al file indicizzatore.txt tutti i nuovi punti dell'immagine			
			for(int r=0; r<16; ++r){
				for(int g=0; g<16; ++g){
					for(int b=0; b<16; ++b){
						if(cube[r][g][b].n_linea>0) {
							fprintf(f, "%d, %d, %d, %f\n", get_pindex(r,g,b), id_immagine, cube[r][g][b].n_linea, cube[r][g][b].peso);
						}
					}
				}
			}
			
			add_path(img.get_url());
			save_new_id(id_immagine);
			
			
			fclose(f);

			
		}
		
		//ritorna la struttura organizzata salvata in indicizzatore.txt
		//p_immagini[punto] contiene la lista delle immagini che hanno quel punto 
		//TODO ne legge una in più se l'ultima riga è vuota
		vector<list<id_iml> > get_structure() {
			FILE* f = fopen(indicizz_path.c_str(), "r");
			list<id_iml> lista;
			vector<list<id_iml> > p_immagini(pow(16,3), lista);

			while(!feof(f)) {
				id_iml img;
				int index_punto;
				fscanf(f, "%d, %d, %d, %f", &index_punto, &img.id_immagine, &img.id_linea, &img.peso);
				p_immagini[index_punto].push_back(img);	
			}
			fclose(f);
			return p_immagini;
		}

		immagine carica_immagine(string path) {
			FILE* f = fopen(path.c_str(), "r");
			immagine img;
			while(!feof(f)) {
				punto p1,p2,p3,p4;
				float peso;
				int index_punto;
				fscanf(f, "%f %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", 
					&peso, 
					&p1.r, &p1.g, &p1.b, 
					&p2.r, &p2.g, &p2.b,
					&p3.r, &p3.g, &p3.b,
					&p4.r, &p4.g, &p4.b
					);
				linea_img l(p1, p2, p3, p4, peso);
				img.add_linea(l);
			}
			fclose(f);
			return img;
		}

		vector<indice> indicizza_opt(immagine img, int id_img) {
			//vettore contenente i nuovi indicii cha andrò ad aggiungere all'indicizzatore
			vector<indice> indici;

			vector<linea_img> linee = img.get_linee();
			reset_cube();
			
			int n_linea = 1;
			//per ogni linea dell'immagine calcolo la bezier 
			for(vector<linea_img>::iterator it=linee.begin(); it<linee.end(); ++it) {
				linea_img l = *it;
				float peso = l.get_peso();
			
				list<punto> punti = bezier(l.get_p1(), l.get_p2(), l.get_p3(), l.get_p4());
				for(list<punto>::iterator ij=punti.begin(); ij!=punti.end(); ++ij) {
					punto p_app = *ij;
					int r = p_app.r;
					int g = p_app.g;
					int b = p_app.b;
			
					cube[r][g][b].n_linea = n_linea;
					
					cube[r][g][b].peso = peso;
				}
				++n_linea;
			}
			
			//aggiungo al file indicizzatore.txt tutti i nuovi punti dell'immagine			
			for(int r=0; r<16; ++r){
				for(int g=0; g<16; ++g){
					for(int b=0; b<16; ++b){
						if(cube[r][g][b].n_linea>0) {
							indice ind;
							ind.id_img = id_img;
							ind.n_linea = cube[r][g][b].n_linea;
							ind.peso = cube[r][g][b].peso;
							ind.pindex = get_pindex(r,g,b);

							indici.push_back(ind);
						}
					}
				}
			}

			return indici;
		}

		void save_indici(vector<indice> indici) {
			FILE* f=fopen(indicizz_path.c_str(),"a");
			for(int i=0; i<indici.size(); ++i) {
				fprintf(f, "%d, %d, %d, %f\n", indici[i].pindex, indici[i].id_img, indici[i].n_linea, indici[i].peso);	
			}

			fclose(f);
		}

		void aggiungi_path_immagini(vector<string> paths) {
			for(int i=0; i<paths.size(); ++i) {
				add_path(paths[i].c_str());
			}
		}
};



#endif

