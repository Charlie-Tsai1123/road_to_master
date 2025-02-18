#include <iostream>
#include <map>
using namespace std;

int main() {

    // initialize
    map<string, int> myMap = {
        {"Tom", 1},
        {"Jack", 2},
        {"John", 3}
    };

    // insert
    myMap["Charlie"] = 100;

    auto result = myMap.insert({"Charlie", 90}); // check if insert successfully!

    if(!result.second){
        cout << "Insert fail" << endl;
        cout << "myMap[\"Charlie\"] = " << myMap["Charlie"] << endl;
    }

    cout << "------------------------------" << endl;

    // iteration
    for(auto& s: myMap) {
        cout << s.first << " : " << s.second << endl;
    }

    cout << "------------------------------" << endl;

    // begin end
    cout << "iteration using begin and end" << endl;
    for(auto s = myMap.begin(); s != myMap.end(); s++) {
        cout << s->first << " : " << s->second << endl;
    }

    cout << "------------------------------" << endl;

    // rbegin rend
    cout << "iteration using rbegin and rend" << endl;
    for(auto s = myMap.rbegin(); s != myMap.rend(); s++) {
        cout << s->first << " : " << s->second << endl;
    }

    cout << "------------------------------" << endl;

    // erase
    myMap.erase("Charlie");
    cout << "after myMap.erase(\"Charlie\")" << endl;
    for(auto& s: myMap) {
        cout << s.first << " : " << s.second << endl;
    }

    cout << "------------------------------" << endl;

    return 0;
}