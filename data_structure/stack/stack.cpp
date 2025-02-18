#include <iostream>
using namespace std;
#define MAX_SIZE 100

class Stack {

private:
    int* arr;
    int top;

public:

    Stack() {
        top = -1;
        arr = new int[MAX_SIZE];
    }

    ~Stack() {
        delete[] arr;
    }

    bool isEmpty() {
        return (top == -1) ? true : false;
    }

    void push(int val) {
        if (top >= MAX_SIZE-1) {
            cout << "The stack is full!" << endl;
            return;
        }
        arr[++top] = val;
        return;
    }

    void pop() {
        if (this->isEmpty()) {
            cout << "The stack is empty!" << endl;
            return;
        }
        top--;
    }

    void printStack() {
        cout << "the value in stack: ";
        for(int i=0; i<=top; i++) {
            cout << arr[i] << " ";
        }
        cout << endl;
    }

};

int main() {
    char option;
    bool quit = false;
    int val;
    Stack s;

    while(!quit) {
        cout << "1. push \n2. pop \n3. empty \n4.print stack \nq: quit" << endl;
        cin >> option;
        cin.ignore();

        switch(option) {
            case '1':
                cout << "Enter the value you want to insert to stack: ";
                cin >> val;
                cin.ignore();
                s.push(val);
                break;
            case '2':
                s.pop();
                break;
            case '3':
                if (s.isEmpty()) {
                    cout << "The stack is empty!" << endl;
                }else {
                    cout << "The stack is not empty!" << endl;
                }
                break;
            case '4':
                s.printStack();
                break;
            case 'q':
                quit = true;
                break;
            default:
                break;
        }

        cout << "-----------------------" << endl;
    }
    return 0;
}