#include <iostream>
#include <stack>
using namespace std;

int main() {

    // stack<int> s
    // s.push(val)
    // s.pop()
    // s.top()
    // s.empty()
    // s.size()
    
    stack<int> myStack;
    char option;
    bool quit = false;
    int val;

    while(!quit) {
        cout << "1. push \n2. pop \n3. top \n4. size \nq: quit" << endl;
        cin >> option;
        cin.ignore();

        switch(option) {
            case '1':
                cout << "Enter the value you want to push: ";
                cin >> val;
                cin.ignore();
                myStack.push(val);
                cout << "Push " << val << " successfully!" << endl;
                break;
            case '2':
                if(myStack.empty()) {
                    cout << "stack is empty!" << endl;
                }else {
                    myStack.pop();
                    cout << "Pop successfully!" << endl;
                }
                break;
            case '3':
                if(myStack.empty()) {
                    cout << "stack is empty!" << endl;
                }else {
                    cout << "The top of stack is " << myStack.top() << endl;
                }
                break;
            case '4':
                cout << "The size of stack is " << myStack.size() << endl;
                break;
            case 'q':
                quit = true;
        }

        cout << "----------------------------" << endl;
    }
    return 0;
}