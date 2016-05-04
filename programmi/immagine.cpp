#ifndef IMMAGINI_CPP
#define IMMAGINI_CPP

#include <vector>
#include <iostream>
#include "linea_img.cpp"


class immagine {
	private:
		int id_immagine;
		string url;
		std::vector<linea_img> linee;
		
	public:	
		immagine() {
			
		}
		
		immagine(string url) {
			this->url = url;
		}
		
		void add_linea(linea_img l) {
			linee.push_back(l);
		}
		
		void print_data() {
			std::cout<<"start print data"<<std::endl;
			for(unsigned i=0; i<linee.size(); ++i) {
				//std::cout<<"linea numero"<<i<<std::endl;
				linee[i].print_data();
			}
		}
	
		
		void set_id_immagine(int id_immagine) {
			this->id_immagine = id_immagine;
		}
		
		std::vector<linea_img> get_linee() {
			return linee;
		}
		
		string get_url() {
			return url;
		}
		
		
};	

#endif
