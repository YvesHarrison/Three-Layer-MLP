#include "std_lib_facilities.h"
#include "Numpy.h"
#include "activation.h"
#include "MLP.h"

void read_csv(string &filename, vector< vector<double> > &output)
{
	fstream file(filename, ios::in);

	if (!file.is_open())
	{
		cout << "File not found!\nEnter correct file name: ";
	}

	string line;

	while (getline(file, line))
	{
		istringstream csv_stream(line);
		vector<double> oneline;
		string element;

		while (getline(csv_stream, element, ',')) {
			oneline.push_back(stod(element));
		}

		output.push_back(oneline);
	}

	file.close();
}

int main()try{
	string filename = "../../../input/stock_train_data_20170916().csv";
	vector< vector<double> > A;
	read_csv(filename, A);
	vector< vector<double> > x_train;
	vector< vector<double> > y_train;
	vector< vector<double> > y_test;
	vector< vector<double> > x_test;
	
	for (int i = 0; i < A.size(); i++) {
		vector<double> n;
		vector<double> m;
		for (int j = 0; j < 87; ++j) {
			n.push_back(A[i][j]);
			//cout << A[i][j];
		}
		//cout << endl;
		if (i < 0.7*A.size()) {
			m.push_back(A[i][88]);
			y_train.push_back(m);
			x_train.push_back(n);
		}
		else {
			m.push_back(A[i][88]);
			y_test.push_back(m);
			x_test.push_back(n);
		}

	}
	numpy MA{ x_train };
	numpy MB{ y_train };
	numpy MC{ x_test };
	numpy MD{ y_test };
	MLP test = MLP(55);
	test.train(MA, MB, 200 ,0.0001,false,true);
	test.test(MC,MD);
	test.save("../../../model/model.txt");
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