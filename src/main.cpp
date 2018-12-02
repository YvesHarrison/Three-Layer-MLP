#include "std_lib_facilities.h"
#include "Numpy.h"
#include "activation.h"
#include "MLP.h"

int main()try{
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

	vector<vector<double>> C = {
		{ 1 },
		{ 1 },
		{ 0 },
		{ 0 }
	};

	vector<vector<double>> D = {
		{ 1 ,1 ,0 ,0}
	};

	numpy MA{ A };
	numpy MB{ B };
	numpy MC{ C };
	numpy MD{ D };
	MLP test = MLP(10);
	test.train(MA, MC, 10, 0.01);
	test.save("../../../model/model.txt");
	test.load("../../../model/model.txt");
    keep_window_open();

    return 0;
}
catch (exception&e) {
	cerr << e.what() << "\n";
	keep_window_open();
	return 1;
}
catch (...) {
	cerr << "exception\n";
	keep_window_open();
	return 2;
}