#include <iostream>
#include <vector>
using namespace std;

//typedef = reserved keyword used to create an additional name
//          (alias) for another data type.
//          New identifier for an existing type
//          Helps with readability and reduces typos
//          Use when there is a clear benefit
//          Replaced with 'using' (work better w/ templates)

using text_t = std::string;

typedef vector<pair<string, int>> pairlist_t; //new identifier usually ends with _t
typedef string text_t;

int main() {
    pairlist_t pairlist;

    text_t firstName = "Bro";

    cout << firstName << endl;
    return 0;
}