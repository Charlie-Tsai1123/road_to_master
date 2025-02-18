#include <iostream>
using namespace std;

int main() {
    // fill() = Fills a range of elements with a specified value
    //          fill(begin, end, value)

    const int SIZE = 99;
    string foods[SIZE];

    fill(foods, foods + (SIZE/3), "pizza");
    fill(foods + (SIZE/3), foods + (SIZE/3)*2, "hamburger");
    fill(foods + (SIZE/3)*2, foods + SIZE, "hotdog");

    for (string food: foods) {
        cout << food << endl;
    }

    return 0;
}