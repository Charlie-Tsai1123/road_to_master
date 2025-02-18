#include <iostream>
using namespace std;

namespace first {
    int x = 1;
}

namespace second {
    int x = 2;
}

int main() {
    int x = 0;
    cout << x << endl;
    cout << first::x << endl;

    using namespace second;
    cout << x << endl;
    cout << second::x << endl;
    return 0;
}