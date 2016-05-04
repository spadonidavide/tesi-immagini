#include "../programmi/immagine.cpp"
#include "../programmi/crea_dati_db.cpp"
#include "../programmi/I_manager.cpp"
#include "../programmi/DB_manager.cpp"
#include "../imgprof/test.c"
#include <ctime>


int main() {
	clock_t begin = clock();

	//string path;
	immagine img;
	I_manager imanager;
	DB_manager db_manager;
	int id_img = imanager.read_last_id();
	vector<indice> indici;
	vector<string> paths;
	try {	
		FILE *f_path = fopen("path_immagini_ind.txt", "r");
		char path[255];
		while(!feof(f_path)) {
			fscanf(f_path, "%s\n", path);
			printf("%s\n", path);
			analisi_immagine(path);
			if(strlen(path)>0) {
				string path_app(path);

				path_app.insert(8, "_caricate");
				paths.push_back(path_app);

				img = load_file("bezier.txt", path_app.c_str());
				db_manager.save_image(img);
				++id_img;
				vector<indice> indici_app = imanager.indicizza_opt(img, id_img);
				indici.insert(indici.end(), indici_app.begin(), indici_app.end());
			}
		}
		fclose(f_path);
	}catch (const std::exception &e){
			cerr << e.what() << std::endl;
	}

	//salvo gli indici che ho creato
	imanager.save_indici(indici);
	//salvo il nuovo id immagini corrente
	imanager.save_new_id(id_img);
	//aggiungo i path delle immagini che ho generato
	imanager.aggiungi_path_immagini(paths);

	clock_t end = clock();
 	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

 	cout<<endl<<"tempo:"<<elapsed_secs<<endl;


	return 0;
}