#include <iostream>
using namespace std;
int main() {
    double data[] = {1,2,3,4};
    double sum=0;
    for(int i=0;i<4;i++) sum+=data[i];
    cout << sum/4;
    return 0;
}
