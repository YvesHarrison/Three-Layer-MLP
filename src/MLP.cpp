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
		//cout << "b2_grad" << endl;
		//cout << b2_grad << endl;
		
		numpy hide = hidden;
		hide.reshape(1,this->hidden_layer_size);
	 	w2_grad=w2_grad+(0.0-re)*(1.0-re)*(y_train.position(j,0)-re)*hide;//1*hidden_layer_size
		//cout << "w2_grad" << endl;
		//cout << w2_grad << endl;
		
		numpy hidden_T = hidden*(1-hidden);
		hidden_T.reshape(1, this->hidden_layer_size);//1*hidden_layer_size
		
		
		numpy middle = this->w2*hidden_T;
		
		middle = dot(tmp, middle);
		middle = middle* re*(1.0 - re)*(y_train.position(j, 0) - re);
		middle = 0.0 - middle;
		middle.reshape(this->hidden_layer_size, x_train.columns());

	 	w1_grad=w1_grad+middle;
		//cout << "w1_grad" << endl;
		//cout << w1_grad << endl;
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
	int y_row=x_train.rows();
	if(x_row!=y_row) error("unmatch trainning data");
	
	double low=-sqrt(6.0/(x_train.columns()+this->hidden_layer_size));
	double high=sqrt(6.0/(x_train.columns()+this->hidden_layer_size));

	this->w1=numpy(low,high,this->hidden_layer_size,x_train.columns());
	this->w2=numpy(low,high,1,this->hidden_layer_size);
	this->b1=numpy(low,high,this->hidden_layer_size,1);
	this->b2=numpy(low,high,1,1);

	

	for(int i=0;i<iterate;++i){
		cout << i+1 << "th epoch" << endl;
		this->backprop(x_train,y_train,learning_rate);
	}
	cout << "after train" << endl;
	cout<<this->w1<<endl;
	cout<<this->w2<<endl;
	cout<<this->b1<<endl;
	cout<<this->b2<<endl;
}

bool MLP::prediction(numpy x){
	int row=x.rows();
	x.reshape(row,1);
	numpy layer_input=dot(this->w1,x)+this->b1;
	numpy layer_output= sigmod(layer_input);
	numpy output= sigmod(dot(this->w2,layer_output)+this->b2);
	return (output.position(0,0)>0.5?1:0);
}

double MLP::test(numpy x_test,numpy y_test){
	int cnt=0;
	for(int i=0;i<x_test.rows();++i){
		if(this->prediction(x_test.get_row(i))==y_test.position(i,0)) cnt++;
	}
	return cnt/x_test.rows();
}
