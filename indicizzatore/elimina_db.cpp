//compile with:g++ elimina_db.cpp -o elimina -lpqxx -lpq
#include <string>
#include <iostream>
#include <pqxx/pqxx>

using namespace std;
using namespace pqxx;


int main() {
	try{
		connection C("dbname=immagini user=postgres password=segreto hostaddr=127.0.0.1 port=5432");
		if (C.is_open()) {
			//cout << "Opened database successfully: " << C.dbname() << endl;
		} 
		else {
			cout << "Can't open database" << endl;
		}
		work W(C);
		string sql = "delete from possiede; delete from immagini;delete from linee;delete from punti;";
		
		result R( W.exec( sql ));
		W.commit();
		C.disconnect();
	}catch (const std::exception &e){
		cerr << e.what() << std::endl;
	}
	return 0;
}