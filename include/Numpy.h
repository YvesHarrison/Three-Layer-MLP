#pragma once

#include "std_lib_facilities.h"

class numpy {
private:
    vector<vector<double>> nparray;
    int np_rows;
    int np_columns;

public:
	numpy(){};
    numpy(const vector<vector<double>>& data);
    numpy(const double low,const double high,const int row,const int col);
    numpy(const int row,const int col,double value);

    numpy get_row(int row);
    int rows() const;
    int columns() const;
    double position(int row, int column);

	void reshape(int row,int col);
};

numpy dot(numpy &m1, numpy &m2);
ostream& operator<<(ostream& os, numpy& m);
numpy operator + (numpy &m1, numpy &m2);
numpy operator + (numpy &m1, double m2);
numpy operator / (numpy &m1, int m2);
numpy operator / (numpy &m1, double m2);
numpy operator - (numpy &m1, double m2);
numpy operator - (double m2, numpy &m1);
numpy operator - (numpy &m1, int m2);
numpy operator - (int m2, numpy &m1);
numpy operator - (numpy &m1, numpy &m2);
numpy operator * (numpy &m1, numpy &m2);




