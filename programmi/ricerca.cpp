#include "I_manager.cpp"
#include "DB_manager.cpp"
#include <iostream>
#include "crea_dati_db.cpp"
#include "../imgprof/test.c"
#include "linea_img.cpp"
#include "open_image.cpp"
#include <algorithm>
#include <string>


using namespace std;

int N_CLASSIFICA = 50;
string default_ipath;// = "home/davide/Immagini/immagini/immagini_caricate/";

vector<float> bitmask_pesi;
vector<int> bitmask;

el_cube cube[16][16][16];

void reset_bitmask() {
	for(int i=0; i<bitmask.size();++i){
		bitmask[i] = 0;
	}
	
	for(int i=0; i<bitmask_pesi.size();++i){
		bitmask_pesi[i] = 0;
	}

}

int get_pindex(int r, int g, int b) {
	int indice = r + g * 16 + b * pow(16, 2);
	
	return indice;
}


vector<int> plot_bitmask(int n_match) {
	vector<int> id_img_match;
	float max_peso = 0;
	int id_max_peso = 0;
	cout<<"bitmask:"<<endl;
	for(int i=0; i<bitmask.size(); ++i){
		if(bitmask[i]>0){
			cout<<"bitmask: "<<bitmask[i]<<"   "<<bitmask_pesi[i]<<endl;	
		}
	}
	
	id_img_match.push_back(id_max_peso);
	return id_img_match;
	
}

//scorro il mio cubo, quando trovo un punto dell'immagine che sto cercando vado a prendere 
//la lista di tutte le immagini che hanno quel punto e incremento le bitmask
void ricerca_cube(const vector<list<id_iml> > &p_immagini) {
	float max_match=0;
	for(int r=0; r<16; ++r){
		for(int g=0; g<16; ++g){
			for(int b=0; b<16; ++b){
				if(cube[r][g][b].n_linea==-1) {					
					int pindex = get_pindex(r, g, b);
					list<id_iml> lista = p_immagini[pindex];
					for(list<id_iml>::iterator ij=lista.begin(); ij!=lista.end(); ++ij) {
					
						id_iml img = *ij;
						int index_immagine = img.id_immagine;
						
						//prendo come peso il rapporto tra il peso dell'immagine che sto cercando 
						//e quella che ho salvato in p_immagini(<=1)
						float minp = min(cube[r][g][b].peso, img.peso);
						
						float maxp = max(cube[r][g][b].peso, img.peso);
						
						float peso = pow((minp/maxp), 0.2);
						
						
						bitmask_pesi[index_immagine] += peso;
						++bitmask[index_immagine];
						
					}
				}
			}
		}
	}
	
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


vector<string> get_classifica(I_manager manager) {
	vector<string> classifica;
	vector<int> scarti;
	
	for(int j=0; j<N_CLASSIFICA; ++j) {
		int max = 0;
		int index_max;
		
		for(int i=0; i<bitmask_pesi.size(); ++i) {
			if(bitmask_pesi[i]>max && find(scarti.begin(), scarti.end(), i)==scarti.end()){
				max = bitmask_pesi[i];
				index_max = i; 
			}
		}

		string p = manager.get_image_path(index_max);
		cout<<"path mod "<<p<<endl;
		classifica.insert(classifica.begin(), p);
		scarti.push_back(index_max);
	}
	
	return classifica;
}

//imposta il path dove saranno andate a cercare le immagini
string set_default_ipath() {
	char buf[255];
	FILE *f=fopen("../indicizzatore/base_path.txt", "r");
	fscanf(f, "%s", buf);
	string p(buf);
	p += "Immagini/immagini/immagini_caricate/";	
	fclose(f);
	return p;
}

int main() {
	default_ipath = set_default_ipath();
	I_manager manager;
	vector<list<id_iml> > p_immagini = manager.get_structure();
	
	int count_immagini = manager.read_last_id() + 1;
	
	//aggiungo tanti elementi alle bitmask quante sono le mie immagini 
	for(int i=0; i<count_immagini; ++i) {
		bitmask.push_back(0);
		bitmask_pesi.push_back(0);
		
	}

	
	while(true) {
		string path_immagine;
		cout<<"inserisci il path dell'immagine"<<endl;
		cin>>path_immagine;
		
		if(path_immagine=="esci")
			break;
		if(path_immagine=="ricarica") {
			cout<<"ricarico i dati"<<endl;
					
			p_immagini = manager.get_structure();

			count_immagini = manager.read_last_id() + 1;
	
			for(int i=0; i<count_immagini; ++i) {
				bitmask.push_back(0);
				bitmask_pesi.push_back(0);
			}			
		}		
		else{
			
			path_immagine = default_ipath + path_immagine;

			//faccio l'analisi dell'immagine che mi salva i dati in output.txt
			//analisi_immagini(path_immagine.c_str());
			//immagine img = load_file("img_ricerca.txt", path_immagine);
			analisi_immagine(path_immagine.c_str());
			immagine img = manager.carica_immagine("output.txt");

			vector<linea_img> linee = img.get_linee();
			
			//calcolo il cubo relativo all'immagine che devo cercare
			reset_cube();			
			for(vector<linea_img>::iterator it=linee.begin(); it!=linee.end(); ++it) {
				linea_img l = *it;
				float peso = l.get_peso();
				list<punto> punti = bezier(l.get_p1(), l.get_p2(), l.get_p3(), l.get_p4());
				for(list<punto>::iterator ij=punti.begin(); ij!=punti.end(); ++ij) {
					punto p_app = *ij;
					int r = p_app.r;
					int g = p_app.g;
					int b = p_app.b;
			
					cube[r][g][b].n_linea = -1;
					
					cube[r][g][b].peso = peso;
				}
			}
			
			reset_bitmask();
			ricerca_cube(p_immagini);
			plot_bitmask(1);
			
			
			vector<string> paths = get_classifica(manager);
			
			DB_manager db_manager;
			vector<string> c_paths = db_manager.ordina_classifica(paths);

			for(int i=0; i<c_paths.size();++i) {
				cout<<c_paths[i]<<endl;
				//ho aggiunto due volte nel path consecutivamente la cartella immagini_caricate quindi torno indiero(trucchetto)
				//es: immagini_caricate/../immagini_caricate
				c_paths[i] = c_paths[i];
			}
			

			vector<string> titles;
			for(int i=1; i<=c_paths.size(); ++i) {
				string title = "match " + to_string(i);
				titles.push_back(title);
			}

			
			show_image(c_paths, titles);		
		}
	}

	return 0;
}
