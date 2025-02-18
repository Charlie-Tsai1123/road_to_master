#include <iostream>
using namespace std;

struct student {
    string name;
    double gpa;
    bool enrolled;
};

void printStudent(student student) {
    cout << student.name << endl;
    cout << student.gpa << endl;
    cout << student.enrolled << endl;
}

int main() {
    student student1;
    student1.name = "Spongebob";
    student1.gpa = 3.9;
    student1.enrolled = true;

    student student2;
    student2.name = "Patric";
    student2.gpa = 3.5;
    student2.enrolled = true;

    printStudent(student1);
    printStudent(student2);
    
    return 0;
}