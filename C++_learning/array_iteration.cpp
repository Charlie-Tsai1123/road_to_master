#include <iostream>
using namespace std;

int main() {
    string students[] = {"Spongebob", "Patrick", "Squidward"};

    for (int i = 0; i < sizeof(students)/sizeof(students[0]); i++) {
        cout << students[i] << endl;
    }

    cout << "-----------------------" << endl;

    // foreach loop
    for (string student: students) {
        cout << student << endl;
    }
}