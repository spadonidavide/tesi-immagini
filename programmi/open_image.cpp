//compile with: g++ open_image.cpp -O2 -L/usr/X11R6/lib -lm -lpthread -lX11


#include "CImg.h"
using namespace cimg_library;

string abs_path = "/home/davide/Immagini/";


int show_image(const char* path, string title) {
	CImg<unsigned char> image(path);
	image.resize(300,300);
  	CImgDisplay main_disp(image,title.c_str());//, draw_disp(visu,"Intensity profile");
  	while (!main_disp.is_closed()/* && !draw_disp.is_closed()*/) {
    		main_disp.wait();
	}
	return 0;
}


int show_image(const char* path1, string title1, const char* path2, string title2) {
	CImg<unsigned char> image1(path1);
	CImg<unsigned char> image2(path2);
	image1.resize(300,300);
	image2.resize(300,300);
  	CImgDisplay main_disp(image1,title1.c_str()), draw_disp(image2,title2.c_str());
  	while (!main_disp.is_closed()/* && !draw_disp.is_closed()*/) {
    		main_disp.wait();
	}
	return 0;
}


int show_image(vector<string> paths, vector<string> titles) {
	vector<CImg<unsigned char> > images;
	
	for(int i=0; i<paths.size(); ++i) {
		images.push_back(CImg<unsigned char>(paths[i].c_str()));
		//cout<<paths[i]<<endl;
		images[i].resize(300, 300);
	}
	
	vector<CImgDisplay> displays;
	
	for(int i=0; i<titles.size(); ++i) {
		displays.push_back(CImgDisplay(images[i], titles[i].c_str()));
		//cout<<titles[i]<<endl;
	}
	
	while(!displays[0].is_closed()) {
		for(int i=0; i<paths.size(); ++i)
			displays[i].wait();
	}

	return 0;
}
