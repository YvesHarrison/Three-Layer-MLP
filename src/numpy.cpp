#include "Numpy.h"

numpy::numpy(const vector<vector<double>>& data){
    nparray = data;
    np_rows = nparray.size();
    np_columns = nparray[0].size();
}

numpy::numpy(const double low,const double high,const int row,const int col){
	vector<vector<double>>data(row,vector<double>(col,0.0));
	std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dis(low, high);
	
	for(int i=0;i<row;++i){
		for(int j=0;j<col;++j){
			data[i][j]= dis(gen);
		}
	}

	nparray=data;
	np_rows = nparray.size();
    np_columns = nparray[0].size();
}

numpy::numpy(const int row,const int col,double value){
	vector<vector<double>>data(row,vector<double>(col,value));

	nparray=data;
	np_rows = nparray.size();
    np_columns = nparray[0].size();
}

int numpy::rows() const{
    return np_rows;
}

int numpy::columns() const{
    return np_columns;
}

double numpy::position(int row, int column){
    if (0 > row && np_rows <= row && 0 > column && np_columns <= column)
        error("out of bounds");

    return nparray[row][column];
}

numpy numpy::get_row(int row){
	if (0 > row && np_rows <= row )
       error("out of bounds");

    vector<vector<double>>mid;
    mid.push_back(nparray[row]);
    numpy  res(mid);
    return res;
}

numpy dot(numpy &m1, numpy &m2){
	if(m1.columns()!=m2.rows()) error("unmatch numpy array for dot product");
	vector<vector<double>>data(m1.rows(),vector<double>(m2.columns(),0.0));

	for(int i=0;i<m1.rows();++i){
		for(int j=0;j<m2.columns();++j){
			for(int k=0;k<m1.columns();++k){
				data[i][j]+=m1.position(i,k)*m2.position(k,j);
			}
		}
	}

	numpy res(data);
	return res;
}

void numpy::reshape(int row,int col){
	if(this->rows()*this->columns()!=row*col) error("numpy array can not be reshaped");
	vector<vector<double>>data(row,vector<double>(col,0.0));

	for(int i=0;i<row;++i){
		for(int j=0;j<col;++j){
			data[i][j]=this->position((i*col+j)/this->columns(), (i*col + j) %this->columns());
		}
	}

	this->nparray=data;
	this->np_rows = nparray.size();
    this->np_columns = nparray[0].size();
}

numpy operator + (numpy &m1, numpy &m2) {
	if (m1.rows() != m2.rows() || m1.columns() != m2.columns()) error("unmatch numpy array plus");
	int r = m1.rows(), c = m1.columns();
	vector<vector<double>>a(r, vector<double>(c, 0.0));

	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			a[i][j] = m1.position(i, j) + m2.position(i, j);
		}
	}

	numpy m3(a);
	return m3;
}

numpy operator + (numpy &m1, double m2) {
	int r = m1.rows(), c = m1.columns();
	vector<vector<double>>a(r, vector<double>(c, 0.0));

	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			a[i][j] = m1.position(i, j) + m2;
		}
	}
	// 
	numpy m3(a);
	return m3;
}

numpy operator - (numpy &m1, double m2) {
	int r = m1.rows(), c = m1.columns();
	vector<vector<double>>a(r, vector<double>(c, 0.0));

	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			a[i][j] = m1.position(i, j) - m2;
		}
	}
	// 
	numpy m3(a);
	return m3;
}

numpy operator - (double m2,numpy &m1) {
	int r = m1.rows(), c = m1.columns();
	vector<vector<double>>a(r, vector<double>(c, 0.0));

	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			a[i][j] = m2-m1.position(i, j);
		}
	}
	// 
	numpy m3(a);
	return m3;
}

numpy operator - (numpy &m1, int m2) {
	int r = m1.rows(), c = m1.columns();
	vector<vector<double>>a(r, vector<double>(c, 0.0));

	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			a[i][j] = m1.position(i, j) - m2;
		}
	}
	// 
	numpy m3(a);
	return m3;
}

numpy operator - (int m2, numpy &m1) {
	int r = m1.rows(), c = m1.columns();
	vector<vector<double>>a(r, vector<double>(c, 0.0));

	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			a[i][j] = m2 - m1.position(i, j);
		}
	}
	// 
	numpy m3(a);
	return m3;
}

numpy operator - (numpy &m1, numpy &m2) {
	if (m1.rows() != m2.rows() || m1.columns() != m2.columns()) error("unmatch numpy array subtraction");
	int r = m1.rows(), c = m1.columns();
	vector<vector<double>>a(r, vector<double>(c, 0.0));

	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			a[i][j] = m1.position(i, j) - m2.position(i, j);
		}
	}

	numpy m3(a);
	return m3;
}

numpy operator *(numpy &m1, numpy &m2){
	if(m1.columns()!= m2.columns()||m1.rows()!=m2.rows()) error("unmatch numpy array for star product");
	vector<vector<double>>data(m1.rows(),vector<double>(m1.columns(),0.0));

	for(int i=0;i<m1.rows();++i){
		for(int j=0;j<m1.columns();++j){
			data[i][j]+=m1.position(i,j)*m2.position(i,j);
			
		}
	}

	numpy res(data);
	return res;
}

numpy operator *(double m1, numpy &m2) {
	vector<vector<double>>data(m2.rows(), vector<double>(m2.columns(), 0.0));

	for (int i = 0; i < m2.rows(); ++i) {
		for (int j = 0; j < m2.columns(); ++j) {
		data[i][j] += m1*m2.position(i, j);		
		}
	}

	numpy res(data);
	return res;
}

numpy operator *(numpy &m1, double m2) {
	vector<vector<double>>data(m1.rows(), vector<double>(m1.columns(), 0.0));

	for (int i = 0; i < m1.rows(); ++i) {
		for (int j = 0; j < m1.columns(); ++j) {
			data[i][j] += m2 * m1.position(i, j);
		}
	}

	numpy res(data);
	return res;
}

numpy operator / (numpy &m1, int m2) {
	int r = m1.rows(), c = m1.columns();
	vector<vector<double>>a(r, vector<double>(c, 0.0));

	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			a[i][j] = m1.position(i, j) / m2;
		}
	}
	// 
	numpy m3(a);
	return m3;
}

numpy operator / (numpy &m1, double m2) {
	int r = m1.rows(), c = m1.columns();
	vector<vector<double>>a(r, vector<double>(c, 0.0));

	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			a[i][j] = m1.position(i, j) / m2;
		}
	}
	// 
	numpy m3(a);
	return m3;
}
ostream& operator<<(ostream& os, numpy& m){
	//os << "[ ";
    for (int i = 0; i < m.rows(); ++i) {
		//os << "[ ";
        for (int j = 0; j < m.columns(); ++j) {
            os << m.position(i, j) << '\t';
        }
		//os << "] ";
        //if (i < m.rows() - 1) {
            //os << '\n';
        //}
    }
	//os << "] ";
    return os;
}
