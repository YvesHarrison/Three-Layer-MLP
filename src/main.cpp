#include "std_lib_facilities.h"
#include "Numpy.h"
#include "activation.h"
#include "MLP.h"
int main(){
    

    vector<vector<double>> A = {
        { 1, 5, 9, 4, 12 },
        { 7, 12, 3, 5, 8 },
        { 8, 4, 15, 9, 5 },
        { 6, 3, 12, 8, 1 }
    };

    vector<vector<double>> B = {
        { 3, 2, 9, 8 },
        { 1, 2, 3, 4 },
        { 9, 8, 7, 6 },
        { 2, 3, 4, 5 },
        { 9, 7, 5, 3 }
    };

    numpy MA{ A };
    numpy MB{ B };
	cout << "(1-MA):" << '\n';
    cout << MA-MA << '\n';
	cout << "MB:" << '\n';
    cout << MB << '\n';
	numpy MC=dot(MA,MB);
    cout << MC << '\n';
	MB.reshape(4, 5);
	cout << MB << '\n';
	numpy MD = numpy(-0.5, 0.5, 4, 4);
	cout << MD << '\n';
	numpy ME = sigmod(MD);
	cout << "ME" << '\n';
	cout << ME << '\n';
	cout << "1" << '\n';
	numpy MF = ME.get_row(1);
	cout <<  MF<< '\n';
    keep_window_open();

    return 0;
}
