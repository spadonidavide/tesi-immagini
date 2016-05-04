#ifndef LINEA_CPP
#define LINEA_CPP

#include <list>
#include <vector>
#include <iostream>
#include "bezier.cpp"

class linea_img{
	private:
		float peso;
		punto p1;
		punto p2;
		punto p3;
		punto p4;
		int id_linea;
		
	public:
		linea_img(punto p1, punto p2, punto p3, punto p4, float peso) {
			this->p1 = p1;
			this->p2 = p2;
			this->p3 = p3;
			this->p4 = p4;
			this->peso = peso;
		}
		
		void print_data() {			
			std::cout<<peso<<", "<<p1.r<<", "<<p1.g<<", "<<p1.b<<", ";
			std::cout<<p2.r<<", "<<p2.g<<", "<<p2.b<<", ";
			std::cout<<p3.r<<", "<<p3.g<<", "<<p3.b<<", ";
			std::cout<<p4.r<<", "<<p4.g<<", "<<p4.b<<endl;
			/*
			std::cout<<"p2:"<<std::endl;
			std::cout<<"	r:"<<p2.r<<std::endl;
			std::cout<<"	g:"<<p2.g<<std::endl;
			std::cout<<"	b:"<<p2.b<<std::endl;
			
			std::cout<<"p3:"<<std::endl;
			std::cout<<"	r:"<<p3.r<<std::endl;
			std::cout<<"	g:"<<p3.g<<std::endl;
			std::cout<<"	b:"<<p3.b<<std::endl;
			
			std::cout<<"p4:"<<std::endl;
			std::cout<<"	r:"<<p4.r<<std::endl;
			std::cout<<"	g:"<<p4.g<<std::endl;
			std::cout<<"	b:"<<p4.b<<std::endl;
			*/
		}
		
		void set_id_linea(int id_linea) {
			this->id_linea = id_linea;
		}
		
		list<punto> get_bezier() {
			return bezier(p1,p2,p3,p4);
		} 
		
		float get_peso(){
			return peso;
		}
		
		punto get_p1() {
			return p1;
		}
		
		punto get_p2() {
			return p2;
		}
		
		punto get_p3() {
			return p3;
		}
		
		punto get_p4() {
			return p4;
		}
};


#endif


