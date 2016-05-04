#include "../programmi/immagine.cpp"
#include "../programmi/crea_dati_db.cpp"
#include "../programmi/I_manager.cpp"
#include "../programmi/DB_manager.cpp"
#include "test.c"


int main(int argc , char  *argv[]) {
	//string path;
	immagine img;
	I_manager imanager;
	DB_manager db_manager;
	try {
		cout<<argv[1]<<endl;
			
		analisi_immagine(argv[1]);
		string path(argv[1]);
		
		//aggiorno il path dell'immagine siccome viene spostata nella cartella immagini_caricate
		path.insert(8, "_caricate");
		img = load_file("bezier.txt", path.c_str());
		
		imanager.indicizza(img);
		
		db_manager.save_image(img);
			

	}catch (const std::exception &e){
			cerr << e.what() << std::endl;
	}
	
	return 0;
	
	
}