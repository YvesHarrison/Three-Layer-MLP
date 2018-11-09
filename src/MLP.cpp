#include "MLP.h"

MLP::MLP(int layer_size){
	this->hidden_layer_size=layer_size;
}

void MLP::backprop(numpy x_train, numpy y_train, double learning_rate){
	// numpy w1_grad=numpy(this->hidden_layer_size,x_train.columns(),0.0),
	// numpy w2_grad=numpy(1,this->hidden_layer_size,0.0),
	// numpy b1_grad=numpy(this->hidden_layer_size,1,0.0),
	// numpy b2_grad=numpy(1,1,0.0);
	// int row=x_train.rows();
	// for(int j=0;j<row;++j){
	// 	numpy tmp=x.get_row(j)
	// 	numpy hidden=sigmod(dot(this->w1,tmp.reshape(row,1))+this->b1);
	// 	numpy res=sigmod(dot(this->w2,hidden)+b2);

	// 	b2_grad+=(0.0-res)*(1.0-res)*(y_train.get_row(j)-res);
	// 	w2_grad+=(0.0-res)*(1.0-res)*(y_train.get_row(j)-res)*hidden;
	//numpy hidden_T=(1.0-hidden).reshape(1,this->hidden_layer_size)
	// 	w1_grad+=(0.0-dot(tmp,this->w2*dot(this->w2,hidden*hidden_T)))*res*(1.0-res);
	// 	b1_grad+=(0.0-this->w2)*hidden*hidden_T*res*(1.0-res)*(y_train.get_row(j)-res);
	// }
	// this->w1-=learning_rate*w1_grad/row;
	// this->w2-=learning_rate*w2_grad/row;
	// this->b1-=learning_rate*b1_grad/row;
	// this->b2-=learning_rate*b2_grad/row;
}

void MLP::train(numpy x_train,numpy y_train,numpy,int iterate, double learning_rate) {
	double low=-sqrt(6.0/(x_train.columns()+this->hidden_layer_size));
	double high=sqrt(6.0/(x_train.columns()+this->hidden_layer_size));
	this->w1=numpy(low,high,this->hidden_layer_size,x_train.columns());
	this->w2=numpy(low,high,1,this->hidden_layer_size);
	this->b1=numpy(low,high,this->hidden_layer_size,1);
	this->b2=numpy(low,high,1,1);

	for(int i=0;i<iterate;++i){
		this->backprop(x_train,y_train,learning_rate);
	}
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
