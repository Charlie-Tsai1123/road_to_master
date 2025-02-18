#include <iostream>
using namespace std;

int main() {
    string name = "Charlie Tsai";
    cout << "size of string : " << sizeof(name) << endl;

    double gpa = 4.0;
    cout << "size of double : " << sizeof(gpa) << endl;

    char grade = 'A';
    cout << "size of char : " << sizeof(grade) << endl;

    bool student = true;
    cout << "size of student : " << sizeof(student) << endl;

    char grades[] = {'A', 'B', 'C', 'D', 'F'};
    cout << "size of char[] : " << sizeof(grades) << endl;
    cout << "number of the array : " << sizeof(grades)/sizeof(char) << endl;
}