#include <iostream>
#include <vector>
using namespace std;

int main(){
    vector<double> data = {100,102,101,105,110};
    vector<double> w;
    double sum = 0;
    for(int i=0;i<data.size();i++){
        double val = (i+1);
        w.push_back(val);
        sum += val;
    }
    cout << "C++ attention: ";
    for(double &v : w){
        cout << v/sum << " ";
    }
    cout << endl;
    return 0;
}
