#pragma once

#include "std_lib_facilities.h"
#include "Numpy.h"
#include "activation.h"

class MLP{
	int hidden_layer_size;
	numpy w1;
	numpy w2;
	numpy b1;
	numpy b2;
public:
	MLP(){};
	MLP(int layer_size);

	void train();
	bool prediction();
	void baclprop();
}





