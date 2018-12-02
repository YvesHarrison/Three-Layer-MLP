#include "MLP.h"

MLP::MLP(int layer_size){
	this->hidden_layer_size=layer_size;
}

void MLP::backprop(numpy x_train, numpy y_train, double learning_rate){
	numpy w1_grad = numpy(this->hidden_layer_size, x_train.columns(), 0.0);
	numpy w2_grad = numpy(1, this->hidden_layer_size, 0.0);
	numpy b1_grad = numpy(this->hidden_layer_size, 1, 0.0);
	numpy b2_grad=numpy(1,1,0.0);

	int row=x_train.rows();
	
	for(int j=0;j<row;++j){
	 	numpy tmp=x_train.get_row(j);
		tmp.reshape(x_train.columns(), 1);
		
	 	numpy hidden=sigmod(dot(this->w1,tmp)+this->b1);//hidden_layer_size*1
	 	numpy res=sigmod(dot(this->w2,hidden)+b2);//1*1
		
		double re=res.position(0,0);//clear
		
		b2_grad=b2_grad+(0.0-re)*(1.0-re)*(y_train.position(j,0)-re);//1*1
		
		numpy hide = hidden;
		hide.reshape(1,this->hidden_layer_size);
	 	w2_grad=w2_grad+(0.0-re)*(1.0-re)*(y_train.position(j,0)-re)*hide;//1*hidden_layer_size
		
		numpy hidden_T = hidden*(1-hidden);
		hidden_T.reshape(1, this->hidden_layer_size);//1*hidden_layer_size
		
		
		numpy middle = this->w2*hidden_T;
		
		middle = dot(tmp, middle);
		middle = middle* re*(1.0 - re)*(y_train.position(j, 0) - re);
		middle = 0.0 - middle;
		middle.reshape(this->hidden_layer_size, x_train.columns());

	 	w1_grad=w1_grad+middle;
		numpy temp= (0.0 - this->w2)*hidden_T*re*(1.0 - re)*(y_train.position(j, 0) - re);
		temp.reshape(this->hidden_layer_size, 1);
	 	b1_grad=b1_grad+temp;
	}
	
	 this->w1=this->w1-learning_rate*w1_grad/row;
	 this->w2=this->w2-learning_rate*w2_grad/row;
	 this->b1=this->b1-learning_rate*b1_grad/row;
	 this->b2=this->b2-learning_rate*b2_grad/row;
}

void MLP::train(numpy x_train,numpy y_train,int iterate, double learning_rate) {
	int x_row=x_train.rows();
	int y_row=y_train.rows();
	
	if(x_row!=y_row) error("unmatch trainning data");
	
	double low=-sqrt(6.0/(x_train.columns()+this->hidden_layer_size));
	double high=sqrt(6.0/(x_train.columns()+this->hidden_layer_size));

	this->w1=numpy(low,high,this->hidden_layer_size,x_train.columns());
	this->w2=numpy(low,high,1,this->hidden_layer_size);
	this->b1=numpy(low,high,this->hidden_layer_size,1);
	this->b2=numpy(low,high,1,1);

	cout << "Train Begin"<< endl;

	for(int i=0;i<iterate;++i){
		cout << i+1 << "th epoch " ;
		this->backprop(x_train,y_train,learning_rate);
		double res = this->test(x_train, y_train);
		cout << "Training Accuracy: " <<res*100<<"%"<< endl;
	}
	cout << "Training Finished" << endl;

}

double MLP::prediction(numpy x){
	int col=x.columns();
	x.reshape(col,1);
	numpy layer_input=dot(this->w1,x)+this->b1;
	numpy layer_output= sigmod(layer_input);
	numpy output= sigmod(dot(this->w2,layer_output)+this->b2);
	
	return (output.position(0,0)>0.5?1.0:0.0);
}

double MLP::test(numpy x_test,numpy y_test){
	int cnt=0;
	for(int i=0;i<x_test.rows();++i){
		if (this->prediction(x_test.get_row(i)) == y_test.position(i, 0))cnt++;
	}
	return (double) cnt/x_test.rows();
}

void MLP::save(string filename) {
	if (filename == "")error("Saving file name must be provided!");
	ofstream fout(filename);
	fout << "hidden_layer_size" << endl;
	fout << this->hidden_layer_size<<endl;
	fout << "w1" << endl;
	fout << this->w1 << endl;
	//cout << "w1" << endl;
	//cout << w1 << endl;
	fout << "b1" << endl;
	fout << this->b1 << endl;
	//cout << "b1" << endl;
	//cout << b1 << endl;
	fout << "w2" << endl;
	fout << this->w2 << endl;
	//cout << "w2" << endl;
	//cout << w2 << endl;
	fout<<"b2"<<endl;
	fout << this->b2 << endl;
	//cout << "b2" << endl;
	//cout << b2 << endl;
	fout.close();
	cout << "Save finished" << endl;
}

void MLP::load(string filename) {
	ifstream inFile(filename, ios::in);
	string lineStr;
	string pre;
	int hidden = -1;
	vector<vector<double>>wt1;
	vector<vector<double>>wt2;
	vector<vector<double>>bt1;
	vector<vector<double>>bt2;
	numpy n_w1;
	numpy n_b1;
	numpy n_w2;
	numpy n_b2;
	while (getline(inFile, lineStr)) {
		stringstream ss(lineStr);
		//cout << lineStr << endl;
		string str;
		vector<string> line;
		while (getline(ss, str, '\t')) {
			line.push_back(str);
			//cout << str << " ";
		}
		if (pre == "hidden_layer_size") {
			hidden = stoi(line[0]);
		}
		else if (pre == "w1") {
			cout << "Loading w1..." << endl;
			vector<double>s1;
			for (int i = 0; i < line.size(); ++i) {
				s1.push_back(stod(line[i]));
			}

			if (hidden == -1)error("Wrong input format");
			wt1.push_back(s1);
			n_w1 = numpy(wt1);
			n_w1.reshape(hidden, wt1[0].size() / hidden);
			
		}
		else if (pre == "b1") {
			cout << "Loading b1..." << endl;
			vector<double>s2;
			for (int i = 0; i < line.size(); ++i) {
				s2.push_back(stod(line[i]));
			}

			if (hidden == -1)error("Wrong input format");
			bt1.push_back(s2);
			n_b1 = numpy(bt1);
			n_b1.reshape(hidden, 1);
			
		}
		else if (pre == "w2") {
			cout << "Loading w2..." << endl;
			vector<double>s3;
			for (int i = 0; i < line.size(); ++i) {
				s3.push_back(stod(line[i]));
			}

			if (hidden == -1)error("Wrong input format");
			wt2.push_back(s3);
			n_w2 = numpy(wt2);
			n_w2.reshape(1, hidden);
		}
		else if (pre == "b2") {
			cout << "Loading b2..." << endl;
			vector<double>s4;
			for (int i = 0; i < line.size(); ++i) {
				s4.push_back(stod(line[i]));
			}

			if (hidden == -1)error("Wrong input format");
			bt2.push_back(s4);
			n_b2 = numpy(bt2);
			n_b2.reshape(1, 1);
		}
		pre = lineStr;
	}
	this->hidden_layer_size = hidden;
	this->w1 = n_w1;
	this->w2 = n_w2;
	this->b1 = n_b1;
	this->b2 = n_b2;
	
	//cout << "w1" << endl;
	//cout << this->w1 << endl;
	
	//cout << "b1" << endl;
	//cout << this->b1 << endl;
	
	//cout << "w2" << endl;
	//cout << this->w2 << endl;
	
	//cout << "b2" << endl;
	//cout << this->b2 << endl;
}

