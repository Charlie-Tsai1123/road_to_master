#include <iostream>
using namespace std;

struct Node {
    int val;
    Node* next;

    Node() : val(0), next(nullptr) {}
    Node(int val) : val(val), next(nullptr) {}
};

class Queue {
    Node *front, *rear;

public:
    Queue() : front(nullptr), rear(nullptr) {}

    bool isEmpty() {
        return (front == nullptr) ? true : false;
    }

    void enQueue(int val) {
        Node* newNode = new Node(val);

        if(this->isEmpty()) {
            front = rear = newNode;
            return;
        }

        rear->next = newNode;
        rear = newNode;
        return;
    }

    void deQueue() {
        if(this->isEmpty()) {
            cout << "underflow!" << endl;
            return;
        }

        Node* temp = front;
        front = front->next;
        delete temp;
        if(front == nullptr) rear = nullptr;
        return;
    }

    int getFront() {
        if(this->isEmpty()) {
            cout << "underflow!" << endl;
            return -1;
        }

        return front->val;
    }

    int getRear() {
        if(this->isEmpty()) {
            cout << "underflow!" << endl;
            return -1;
        }
        return rear->val;
    }

    void printQueue() {
        Node* temp = front;
        while(temp != nullptr) {
            cout << temp->val << " -> ";
            temp = temp->next;
        }
        cout << "nullptr" << endl;
        return;
    }
};

int main() {
    char option;
    bool quit = false;
    Queue myQueue = Queue();
    int val;

    while(!quit) {
        cout << "1. enQueue \n2. deQueue \n3. getFront \n4. getRear \n5. traversal \nq: quit" << endl;
        cin >> option;
        cin.ignore();

        switch(option) {
            case '1':
                cout << "Enter the number you want to insert to queue: ";
                cin >> val;
                cin.ignore();
                myQueue.enQueue(val);
                break;
            case '2':
                myQueue.deQueue();
                break;
            case '3':
                val = myQueue.getFront();
                if(val != -1) cout << "The front value is " << val << endl;
                break;
            case '4':
                val = myQueue.getRear();
                if(val != -1) cout << "The rear value is " << val << endl;
                break;
            case '5':
                myQueue.printQueue();
                break;
            case 'q':
                quit = true;
                break;
            default:
                break;
        }

        cout << "-----------------------------" << endl;
    }
    return 0;
}