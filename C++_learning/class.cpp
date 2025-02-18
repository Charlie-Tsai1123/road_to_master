#include <iostream>
using namespace std;

class Human {
    public:
        string name;
        string occupation;
        int age;

        void eat() {
            cout << "This person is eating" << endl;
        }

        void drink() {
            cout << "This person is drinking" << endl;
        }

        void sleep() {
            cout << "This person is sleeping" << endl;
        }
};

int main() {
    
    Human human1;

    human1.name = "Rick";
    human1.occupation = "scientist";
    human1.age = 70;

    cout << human1.name << endl;
    cout << human1.occupation << endl;
    cout << human1.age << endl;

    human1.eat();
    human1.drink();
    human1.sleep();

    return 0;
}