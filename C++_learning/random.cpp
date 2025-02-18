#include <iostream>
#include <ctime>
using namespace std;

int main() {
    // pseudo-random = NOT truly random (but close)

    srand(time(NULL));

    int num = rand(); // 0 ~ 32767
    cout << num;
    return 0;
}