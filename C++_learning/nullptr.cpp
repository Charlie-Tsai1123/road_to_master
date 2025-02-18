#include <iostream>
using namespace std;

int main() {
    int age = 20;
    int* pAge = nullptr;

    pAge = &age;
    cout << "The address of age is " << pAge << endl;
    cout << "The value of age is " << *pAge << endl;
}