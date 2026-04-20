#include <iostream>
using namespace std;
int main() {
    double data[5] = {100,102,101,105,110};
    double sum = 0;
    for(int i=0;i<5;i++) sum += data[i];
    cout << "Result: " << sum/5 << endl;
    return 0;
}
