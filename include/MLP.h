#pragma once

#include "std_lib_facilities.h"
#include "Numpy.h"
#include "activation.h"

class MLP{
public:
	int hidden_layer_size;
	numpy w1;
	numpy w2;
	numpy b1;
	numpy b2;
	MLP(){};
	MLP(int layer_size);

	void train(numpy &x_train,numpy &y_train,int iterate, double learning_rate,bool adjust, bool save);
	double prediction(numpy x);
	void backprop(numpy &x_train, numpy &y_train,double learning_rate);
	double test(numpy &x_test,numpy &y_test);
	void save(string filename);
	void load(string filename);
};
