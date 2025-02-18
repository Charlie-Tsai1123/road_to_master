#include <iostream>
using namespace std;

class Student{
    public:
        string name;
        int age;
        double gpa;

    Student(string name, int age, double gpa) {
        this->name = name;
        this->age = age;
        this->gpa = gpa;
    }
};

int main() {

    // constructor = special method that is automatically called when an object is instantiated
    //               useful for assigning values to attributes as arguments 

    Student student1("Spongebob", 25, 3.2);

    cout << student1.name << endl;
    cout << student1.age << endl;
    cout << student1.gpa << endl;
    return 0;
}