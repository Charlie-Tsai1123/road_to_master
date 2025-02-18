#include <iostream>
using namespace std;

int main() {
    string name;
    int age;

    cout << "What's your age?: ";
    cin >> age;

    cout << "What's your full name?: ";
    //getline(cin, name); // fail because it read \n (previous cin doesn't read \n)
    getline(cin >> ws, name); // ignore begin '\n'

    cout << "Hello " << name << '\n';
    cout << "Your are " << age << " years old" << endl;

    // cin error method
    cout << "This is cin.clear() example" << endl;
    int num;
    do {
        cin >> num;
        cin.clear();
        fflush(stdin);
        cout << num << endl;
    } while (num != -1);

    return 0;
}