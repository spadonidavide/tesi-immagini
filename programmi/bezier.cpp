#ifndef BEZIER_CPP
#define BEZIER_CPP
#include <list>

#include <vector>

#include <iostream>

#include <math.h>


using namespace std;

int N_PUNTI=15;
int MAX_LIMIT = 15;
int MIN_LIMIT = 0;

struct punto{
	
	int r;
	int g;
	int b;
	
	bool operator==(punto p1){
		if(p1.r!=r)
			return false;
		if(p1.g!=g)
			return false;
		if(p1.b!=b)
			return false;
		return true;
	}

};

int sat(int n) {
	if(n>15)
		return 15;
	
	if(n<0)
		return 0;
	
	return n;
}	

list<punto> bezier(punto p1, punto p2, punto p3, punto p4) {
	float x[N_PUNTI];
	float y[N_PUNTI];
	float z[N_PUNTI];
	list<punto> punti;
	
	
	for(int i=1; i<=N_PUNTI; ++i) {
		float t = i/((1.0*N_PUNTI));
		x[i-1] = sat(round(p1.r * pow((1-t),3) + 3 * p2.r * t * pow((1-t), 2) + 3 * p3.r * pow(t,2) * (1-t) + p4.r * pow(t,3)));
		y[i-1] = sat(round(p1.g * pow((1-t),3) + 3 * p2.g * t * pow((1-t), 2) + 3 * p3.g * pow(t,2) * (1-t) + p4.g * pow(t,3)));
		z[i-1] = sat(round(p1.b * pow((1-t),3) + 3 * p2.b * t * pow((1-t), 2) + 3 * p3.b * pow(t,2) * (1-t) + p4.b * pow(t,3)));
		
		
	}
	
	
	for(int i=0; i<N_PUNTI; ++i) {
		punto p;
		p.r = x[i];
		p.g = y[i];
		p.b = z[i];
		
		punti.push_back(p);	;
		
		
		
	}
	punti.unique();
	return punti;
	

}

#endif

	/*
	cout<<"x = [";
	for(int i=0; i<N_PUNTI; ++i) {
		cout<<x[i]<<" ";
	}
	cout<<"]"<<endl;
	
	cout<<"y = [";
	for(int i=0; i<N_PUNTI; ++i) {
		cout<<y[i]<<" ";
	}
	cout<<"]"<<endl;

	cout<<"z = [";
	for(int i=0; i<N_PUNTI; ++i) {
		cout<<z[i]<<" ";
	}
	cout<<"]"<<endl;*/



/*
int main() {
	punto p1,p2,p3,p4;
	p1.r = 1;
	p1.g = 2;
	p1.b = 3;
	
	p2.r = 1;
	p2.g = 5;
	p2.b = 3;
	
	p3.r = 8;
	p3.g = 2;
	p3.b = 3;
	
	p4.r = 1;
	p4.g = 9;
	p4.b = 3;
	
	bezier(p1,p2,p3,p4);
	return 0;
}*/





