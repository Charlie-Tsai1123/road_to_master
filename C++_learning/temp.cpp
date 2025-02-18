#include <iostream>
using namespace std;

class Solution {
    public:

        string s;

        string longestPalindrome() {
            string palindrome = "";
            int max_length = 0;
            for(int i = 0; i < s.length(); i++) {
    
                for(int j = max_length + 1; j <= s.length(); j++) {
                    string current = s.substr(i, j);
                    
                    if(isPalindrome(current)) {
                        palindrome = current;
                        max_length = j;
                    }

                    cout << current << " " << palindrome << " " << max_length << endl;
                }
            }
    
            return palindrome;
        }
    
        bool isPalindrome(string s) {
            int left = 0;
            int right = s.length() - 1;
    
            while(left < right) {
                if(s[left] != s[right]) return false;
                left++;
                right--;
            }
            return true;
        }
    };

int main() {
    Solution test;
    cin >> test.s;
    cin.ignore();

    cout << test.longestPalindrome() << endl;
    
}
