#include <iostream>
using namespace std;

struct ListNode {
    int val;
    ListNode* next;

    ListNode() : val(0), next(nullptr) {}
    ListNode(int val) : val(val), next(nullptr) {}
};

class LinkedList {
    ListNode* head;

public:
    LinkedList() : head(nullptr) {}

    void printList() {
        ListNode* temp = head;
        while(temp != nullptr) {
            cout << temp->val << " -> ";
            temp = temp->next;
        }

        cout << "nullptr" << endl;
        return;
    }

    void insertAtEnd(int val) {
        ListNode* newNode = new ListNode(val);

        if(head == nullptr) {
            head = newNode;
            return;
        }

        ListNode* current = head;
        while(current->next != nullptr) current = current->next;
        current->next = newNode;
        return;
    }

    void insertAtHead(int val) {
        ListNode* newNode = new ListNode(val);
        newNode->next = head;
        head = newNode;
        return;
    }

    void deleteNode(int val) {
        if(head == nullptr) return;

        if(head->val == val) {
            ListNode* temp = head;
            head = head->next;
            delete temp;
            cout << "delete " << val << " successfully!" << endl;
            return;
        }

        ListNode* current = head;
        while(current->next != nullptr && current->next->val != val) current = current->next;
        if(current->next != nullptr) {
            ListNode* temp = current->next;
            current->next = current->next->next;
            delete temp;
            cout << "delete " << val << " successfully!" << endl;
        }
        return;
    }

    void reverseList() {
        ListNode* previous = nullptr;
        ListNode* current = head;
        while(current != nullptr) {
            ListNode* next = current->next;
            current->next = previous;
            previous = current;
            current = next;
        }
        head = previous;
        return;
    }

    void freeList() {
        while(head != nullptr) {
            ListNode* temp = head;
            head = head->next;
            delete temp;
        }
        return;
    }
};

int main() {
    char option;
    bool quit = false;
    int val;
    LinkedList head = LinkedList();

    while(!quit) {
        cout << "1. print list \n2. insert at end \n3. insert at head \n4. delete node \n5. reverse linked list \n6. free list \nq: quit" << endl;
        cin >> option;
        cin.ignore();

        switch(option) {
            case '1':
                head.printList();
                break;
            case '2':
                cout << "Enter the value you want to insert at end: ";
                cin >> val;
                cin.ignore();
                head.insertAtEnd(val);
                break;
            case '3':
                cout << "Enter the value you want to insert at head: ";
                cin >> val;
                cin.ignore();
                head.insertAtHead(val);
                break;
            case '4':
                cout << "Enter the value you want to delete: ";
                cin >> val;
                cin.ignore();
                head.deleteNode(val);
                break;
            case '5':
                head.reverseList();
                break;
            case '6':
                head.freeList();
                break;
            case 'q':
                quit = true;
                break;
            default:
                break;
        }

        cout << "----------------------------" << endl;
    }
    return 0;
}