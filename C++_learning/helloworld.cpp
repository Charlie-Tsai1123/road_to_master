#include <iostream>
using namespace std;

int main() {
    // This is comment
    /* comment with multiple line
    line
    line
    */
    cout << "hello world!" << endl;

    int x; //declaration
    x = 5; //assignment
    cout << x << endl;


    // The const keyword specifies tthat a variable's value is constant
    // tells the compiler to prevent anything from modifying it
    // (read-only)
    const double PI = 3.14159;
    double radius = 10;
    double circumference = 2 * PI * radius;
    cout << circumference << "cm";
    
    

    return 0;
}