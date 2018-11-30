#pragma once

#include "std_lib_facilities.h"
#include "Numpy.h"

numpy sigmod(numpy& x);
numpy tanh(numpy& x);
numpy ReLU(numpy& x);
numpy ELU(numpy& x,double alpha);
numpy PReLU(numpy& x, double alpha);
double sigmod(double x);
double tanh(double x);
double ReLU(double x);
double ELU(double x, double alpha);
double PReLU(double x, double alpha);




