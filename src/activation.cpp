#include "activation.h"

//activate functions use numpy object as input parameter
numpy sigmod(numpy& x){
	vector<vector<double>>data(x.rows(),vector<double>(x.columns(),0.0));

	for(int i=0;i<x.rows();++i){
		for(int j=0;j<x.columns();++j){
			data[i][j]=1/(1+exp(-x.position(i,j)));
		}
	}

	numpy res(data);
	return res;
}

numpy tanh(numpy& x){
	vector<vector<double>>data(x.rows(),vector<double>(x.columns(),0.0));

	for(int i=0;i<x.rows();++i){
		for(int j=0;j<x.columns();++j){
			data[i][j]=(exp(x.position(i,j))-exp(-x.position(i,j)))/(exp(x.position(i,j))+exp(-x.position(i,j)));
		}
	}

	numpy res(data);
	return res;
}

numpy ReLU(numpy& x){
	vector<vector<double>>data(x.rows(),vector<double>(x.columns(),0.0));

	for(int i=0;i<x.rows();++i){
		for(int j=0;j<x.columns();++j){
			data[i][j]=max(0.0,x.position(i,j));
		}
	}

	numpy res(data);
	return res;
}

numpy ELU(numpy& x,double alpha){
	vector<vector<double>>data(x.rows(),vector<double>(x.columns(),0.0));

	for(int i=0;i<x.rows();++i){
		for(int j=0;j<x.columns();++j){
			data[i][j] = (x.position(i, j) > 0.0 ? x.position(i, j):alpha*(exp(x.position(i, j)) - 1)); 
		}
	}

	numpy res(data);
	return res;
}

numpy PReLU(numpy& x,double alpha){
	vector<vector<double>>data(x.rows(),vector<double>(x.columns(),0.0));

	for(int i=0;i<x.rows();++i){
		for(int j=0;j<x.columns();++j){
			data[i][j]=max(alpha*x.position(i,j),x.position(i,j));
		}
	}

	numpy res(data);
	return res;
}

//activate functions use double as input parameter
double sigmod(double x){
	return 1/(1+exp(-x));
}

double tanh(double x){
	return (exp(x)-exp(-x))/(exp(x)+exp(-x));
}

double ReLU(double x){
	return max(0.0,x);
}

double ELU(double x,double alpha){
	return (x>0?x:alpha*(exp(x)-1));
}

double PReLU(double x,double alpha){
	return max(alpha*x,x);
}