#include <iostream>
using namespace std;

int main() {
    string name;

    cout << "Enter your name: ";
    getline(cin, name);

    cout << "name.length() = " << name.length() << endl;
    cout << "name.empty() = " << name.empty() << endl;

    name.clear();
    cout << "name.clear => name = " << name << endl;

    getline(cin, name);
    name.append("@gmail.com");
    cout << "name.append(\"@gmail.com\") => name = " << name << endl;

    cout << "name.at(0) = " << name.at(0) << endl;

    getline(cin, name);
    name.insert(1, "@");
    cout << "name.insert(1, \"@\") => name = " << name << endl;
    cout << "name.find('i') = " << name.find('i') << endl;

    cout << "name.erase(0, 9) = " << name.erase(0, 9) << endl;
    return 0;
}