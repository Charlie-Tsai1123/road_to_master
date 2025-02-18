#include <iostream>
using namespace std;

void bubble_sort(int array[], int size);

int main() {
    int array[] = {10, 1, 9, 3, 8, 2, 7, 4, 6, 5};
    int size = sizeof(array) / sizeof(array[0]);

    bubble_sort(array, size);

    for (int element: array) {
        cout << element << " ";
    }
    cout << endl;
    return 0;
}

void bubble_sort(int array[], int size) {
    int temp;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size - i; j++) {
            if (array[j] < array[j+1]) continue;
            temp = array[j];
            array[j] = array[j+1];
            array[j+1] = temp;
        }
    }
}