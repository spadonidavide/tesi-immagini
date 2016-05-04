#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

using namespace std;

int n_lines(const char* filename) {
	unsigned int number_of_lines = 0;
	FILE *infile = fopen(filename, "r");
	int ch;

	while (EOF != (ch=getc(infile)))
		if ('\n' == ch)
			++number_of_lines;

	return number_of_lines;	
}


// r,g,b values are from 0 to 255
// h = [0,1], s = [0,1], v = [0,1]
//		if s == 0, then h = -1 (undefined)
#ifndef RGB2HSV1
#define RGB2HSV1
void RGBtoHSV1( float r, float g, float b, float *h, float *s, float *v ) {
	float min, max, delta;
	r/=255;
	g/=255;
	b/=255;
	min = r;
	if (min>g) min=g;
	if (min>b) min=b;
	max = r;
	if (max<g) max=g;
	if (max<b) max=b;

	*v = max;				// v

	delta = max - min;

	if( max != 0 )
		*s = delta / max;		// s
	else {
		// r = g = b = 0		// s = 0, v is undefined
		*s = 0;
		*h = 0;
		return;
	}

	if (delta==0) *h=0;
	else{
		if( r == max )
			*h = ( g - b ) / delta;		// between yellow & magenta
		else if( g == max )
			*h = 2 + ( b - r ) / delta;	// between cyan & yellow
		else
			*h = 4 + ( r - g ) / delta;	// between magenta & cyan

		*h *= 60;				// degrees
		if( *h < 0 )
			*h += 360;
		
		*h/=360;
	}
}
#endif

float match_prof() {
	int S = 16;
	int NB = 256/S;
	
	float total_matcha = 0;
	float total_matchb = 0;

	int n_f2 = n_lines("output2.txt");

	FILE* f1 = fopen("output.txt","r+");
	while ( !feof(f1)) {
	//// compare with bezier

	int bezier_test1[12];
	float w_test1=0;

	fscanf(f1,"%f",&w_test1);

	for (int i=0;i<12;i++)
	fscanf(f1,"%d, ",&bezier_test1[i]);

	//printf("%f: ",w_test1);
	//for (int i=0;i<12;i++)
		//printf("%d, ",bezier_test1[i]);

	//printf("\n");



	float matcha=1000;
	float matchb=1000;


	FILE* f2 = fopen("output2.txt","r+");
	for (int j2=0;j2<n_f2;j2++)
		if (/*lists[j2].size()*/1>0){ // non cancellato

			int bezier_test2[12];
			float w_test2=0;

			fscanf(f2,"%f",&w_test2);

			for (int i=0;i<12;i++)
				fscanf(f2,"%d, ",&bezier_test2[i]);

			//printf("%f: ",w_test2);
			//for (int i=0;i<12;i++)
				//printf("%d, ",bezier_test2[i]);

			//printf("\n");



			int n=20;
			int dbg=0;
			float hsv1[3*n];
			float hsv2[3*n];
			float delta[n][3]; // elenco delta tra coppie match migliori
			float delta1[n][3]; // elenco delta tra coppie match migliori
			float bar[3];      // baricentro delta
			bar[0]=0;
			bar[1]=0;
			bar[2]=0;
			for (int i=0;i<n;i++){
				float r1,g1,b1,r2,g2,b2;
				float h1,h2,s1,s2,v1,v2;
				float t=((float)i)/(n-1);
				float t1=((float)i)/(n-1);
				r1=(pow(1-t,3)*bezier_test1[0]+3*t*(1-t)*(1-t)*bezier_test1[3]+3*t*t*(1-t)*bezier_test1[6]+t*t*t*bezier_test1[9]);
				g1=(pow(1-t,3)*bezier_test1[1]+3*t*(1-t)*(1-t)*bezier_test1[4]+3*t*t*(1-t)*bezier_test1[7]+t*t*t*bezier_test1[10]);
				b1=(pow(1-t,3)*bezier_test1[2]+3*t*(1-t)*(1-t)*bezier_test1[5]+3*t*t*(1-t)*bezier_test1[8]+t*t*t*bezier_test1[11]);
				r2=(pow(1-t1,3)*bezier_test2[0]+3*t1*(1-t1)*(1-t1)*bezier_test2[3]+3*t1*t1*(1-t1)*bezier_test2[6]+t1*t1*t1*bezier_test2[9]);
				g2=(pow(1-t1,3)*bezier_test2[1]+3*t1*(1-t1)*(1-t1)*bezier_test2[4]+3*t1*t1*(1-t1)*bezier_test2[7]+t1*t1*t1*bezier_test2[10]);
				b2=(pow(1-t1,3)*bezier_test2[2]+3*t1*(1-t1)*(1-t1)*bezier_test2[5]+3*t1*t1*(1-t1)*bezier_test2[8]+t1*t1*t1*bezier_test2[11]);
				if (r1<0) r1=0;
				if (g1<0) g1=0;
				if (b1<0) b1=0;
				if (r2<0) r2=0;
				if (g2<0) g2=0;
				if (b2<0) b2=0;
				if (r1>=NB) r1=NB-1;
				if (g1>=NB) g1=NB-1;
				if (b1>=NB) b1=NB-1;
				if (r2>=NB) r2=NB-1;
				if (g2>=NB) g2=NB-1;
				if (b2>=NB) b2=NB-1;	      
				RGBtoHSV1(r1*S,g1*S,b1*S,&h1,&s1,&v1);
				RGBtoHSV1(r2*S,g2*S,b2*S,&h2,&s2,&v2);
				hsv1[3*i+0]=h1;
				hsv1[3*i+1]=s1;
				hsv1[3*i+2]=v1;
				hsv2[3*i+0]=h2;
				hsv2[3*i+1]=s2;
				hsv2[3*i+2]=v2;
			}


			for (int swap=0;swap<2;swap++) { // solo primo test : cerco sample su elenco
				// per ogni punto trova il migliore nell'altro
				float best=0;
				float mindist=1000;
				float maxdist=0;
				char bitmask[n];
				for (int i=0;i<n;i++)
					bitmask[i]=0;

				for (int i=0;i<n;i++) {
					float bestdist=1000;
					for (int j=0;j<n;j++) {
						float h1,h2,s1,s2,v1,v2;
						if (swap==0){
							h1=hsv1[3*i+0];
							s1=hsv1[3*i+1];
							v1=hsv1[3*i+2];
							h2=hsv2[3*j+0];
							s2=hsv2[3*j+1];
							v2=hsv2[3*j+2];
						}else{
							h1=hsv2[3*i+0];
							s1=hsv2[3*i+1];
							v1=hsv2[3*i+2];
							h2=hsv1[3*j+0];
							s2=hsv1[3*j+1];
							v2=hsv1[3*j+2];
						}
						//	    float dist=(r1-r2)*(r1-r2)+(g1-g2)*(g1-g2)+(b1-b2)*(b1-b2);
						float c1x,c1y,c1z;
						float c2x,c2y,c2z;
						c1x=cos(h1*2*3.1415)*s1;
						c1y=sin(h1*2*3.1415)*s1;
						c2x=cos(h2*2*3.1415)*s2;
						c2y=sin(h2*2*3.1415)*s2;
						c1z=v1;
						c2z=v2;
						float disth=pow((c1x-c2x)*(c1x-c2x)+(c1y-c2y)*(c1y-c2y)+(c1z-c2z)*(c1z-c2z),0.5);
						/*
						fabs(h1-h2);
						if (fabs(h1-h2)>0.5)
						disth=1-disth;
						*/


						float dist=disth;
						//if (dbg)
							//printf("    %d %d %f: %f %f %f - %f %f %f\n",i,j,dist,h1,s1,v1,h2,s2,v2);

						if (bestdist>dist){
							bestdist=dist;
							delta[i][0]=c1x-c2x;
							delta[i][1]=c1y-c2y;
							delta[i][2]=c1z-c2z;
						}
					}
				}

				//valuta delta
				//media --> baricentro
				for (int i=0;i<n;i++){
					bar[0]+=delta[i][0];
					bar[1]+=delta[i][1];
					bar[2]+=delta[i][2];
				}
				bar[0]/=n;
				bar[1]/=n;
				bar[2]/=n;

				//ricalcolo distanza
				for (int i=0;i<n;i++) {
					float bestdist=1000;
					int bestidx=-1;
					for (int j=0;j<n;j++) {
						float h1,h2,s1,s2,v1,v2;
						if (swap==0){
							h1=hsv1[3*i+0];
							s1=hsv1[3*i+1];
							v1=hsv1[3*i+2];
							h2=hsv2[3*j+0];
							s2=hsv2[3*j+1];
							v2=hsv2[3*j+2];
						}else{
							h1=hsv2[3*i+0];
							s1=hsv2[3*i+1];
							v1=hsv2[3*i+2];
							h2=hsv1[3*j+0];
							s2=hsv1[3*j+1];
							v2=hsv1[3*j+2];
					}
					//	    float dist=(r1-r2)*(r1-r2)+(g1-g2)*(g1-g2)+(b1-b2)*(b1-b2);
					float c1x,c1y,c1z;
					float c2x,c2y,c2z;
					c1x=cos(h1*2*3.1415)*s1;
					c1y=sin(h1*2*3.1415)*s1;
					c2x=cos(h2*2*3.1415)*s2;
					c2y=sin(h2*2*3.1415)*s2;
					c1z=v1;
					c2z=v2;
					float disth=pow((c1x-c2x-bar[0])*(c1x-c2x-bar[0])+(c1y-c2y-bar[1])*(c1y-c2y-bar[1])+(c1z-c2z-bar[2])*(c1z-c2z-bar[2]),0.5);
					/*
					fabs(h1-h2);
					if (fabs(h1-h2)>0.5)
					disth=1-disth;
					*/


					float dist=disth;
					//if (dbg)
						//printf("    %d %d %f: %f %f %f - %f %f %f\n",i,j,dist,h1,s1,v1,h2,s2,v2);

					if (bestdist>dist){
						bestdist=dist;
						bestidx=j;
						delta1[i][0]=c1x-c2x-bar[0];
						delta1[i][1]=c1y-c2y-bar[1];
						delta1[i][2]=c1z-c2z-bar[2];
					}
				}
				bitmask[bestidx]=1;
				best+=bestdist;
				if (mindist>bestdist) mindist=bestdist;
				if (maxdist<bestdist) maxdist=bestdist;
			}



			float len;
			len=pow(bar[0]*bar[0]+bar[1]*bar[1]+bar[2]*bar[2],0.5f);

			// scarto da baricentro --->
			float scarto=0;
			for (int i=0;i<n;i++){
				scarto+=(bar[0]-delta[i][0])*(bar[0]-delta[i][0])+(bar[1]-delta[i][1])*(bar[1]-delta[i][1])+(bar[2]-delta[i][2])*(bar[2]-delta[i][2]);
			}
			scarto=pow(scarto/n,0.5f);

			//printf("bar %f %f %f\n",bar[0],bar[1],bar[2]);
			//media --> baricentro
			bar[0]=0;
			bar[1]=0;
			bar[2]=0;
			for (int i=0;i<n;i++){
				bar[0]+=delta1[i][0];
				bar[1]+=delta1[i][1];
				bar[2]+=delta1[i][2];
			}
			bar[0]/=n;
			bar[1]/=n;
			bar[2]/=n;

			//printf("bar %f %f %f\n",bar[0],bar[1],bar[2]);

			float scarto1=0;
			for (int i=0;i<n;i++){
				scarto1+=(bar[0]-delta1[i][0])*(bar[0]-delta1[i][0])+(bar[1]-delta1[i][1])*(bar[1]-delta1[i][1])+(bar[2]-delta1[i][2])*(bar[2]-delta1[i][2]);
			}
			scarto1=pow(scarto1/n,0.5f);


			int cover=0;  // per ora non usato
			for (int i=0;i<n;i++){
				//printf("%d",bitmask[i]);
				cover+=bitmask[i];
			}
			//printf("\n");

			float descr=maxdist;
			float descrw;
			if (w_test2>w_test1)	       // rapporto <1 (1 uguale ottimo!)
				descrw=w_test1/w_test2;
			else
				descrw=w_test2/w_test1;

			descrw=pow(descrw,0.2); // schiaccio verso 1 (peso un po' meno le differenze di peso)

			//	    float currmatch=descr/descrw/((float)cover/n); // uso score peggiore (la media non cattura le differenze!)
			float currmatch=descr/descrw; // uso score peggiore (la media non cattura le differenze!)

			if (swap==0)
				if (matcha>currmatch)
					matcha=currmatch;

			if (swap==1)
				if (matchb>currmatch)
					matchb=currmatch;

			//if (swap==0)
				//printf("dis0 %d: %f %f cover %f deltaw %f: est %f dev: %f, %f %f\n",j2,best/n,maxdist,(float)cover/n,descrw,currmatch,len,scarto,scarto1);
			//if (swap==1)
				//printf("dis1 %d: %f %f cover %f deltaw %f: est %f dev: %f, %f %f\n",j2,best/n,maxdist,(float)cover/n,descrw,currmatch,len,scarto,scarto1);
			}
		}
		//printf("score: %f %f\n",matcha,matchb);
		total_matcha+=w_test1*matcha;    // peso con il peso originale della spline
		total_matchb+=w_test1*matchb;    // peso con il peso originale della spline
		//printf("\n");

		fclose(f2);
	} 
	fclose(f1);

	//printf("total score: %f %f %f\n",total_matcha, total_matchb,total_matcha*total_matchb);
	
	return total_matcha*total_matchb;
}

