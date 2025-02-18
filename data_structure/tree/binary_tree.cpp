#include <iostream>
using namespace std;

struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;

    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int val) : val(val), left(nullptr), right(nullptr) {}
};

class BinaryTree {
private: 
    TreeNode* root;

    void preOrder(TreeNode* node) {
        if(node == nullptr) return;

        cout << node->val << " ";
        preOrder(node->left);
        preOrder(node->right);
    }

    void inOrder(TreeNode* node) {
        if(node == nullptr) return;

        inOrder(node->left);
        cout << node->val << " ";
        inOrder(node->right);
    }

    void postOrder(TreeNode* node) {
        if(node == nullptr) return;

        postOrder(node->left);
        postOrder(node->right);
        cout << node->val << " ";
    }

public:

    BinaryTree() : root(nullptr) {
        root = new TreeNode(1);
        root->left = new TreeNode(2);
        root->right = new TreeNode(3);
        root->left->left = new TreeNode(4);
        root->left->right = new TreeNode(5);
        root->right->left = new TreeNode(6);
        root->left->left->left = new TreeNode(7);
        root->left->left->right = new TreeNode(8);
        root->left->right->right = new TreeNode(9);
        root->right->left->right = new TreeNode(10);
    }

    void printPreOrder() {
        preOrder(root);
        cout << endl;
    }

    void printInOrder() {
        inOrder(root);
        cout << endl;
    }

    void printPostOrder() {
        postOrder(root);
        cout << endl;
    }

};

int main() {
    BinaryTree myTree = BinaryTree();
    
    cout << "prefix:" << endl;
    myTree.printPreOrder();

    cout << "infix: " << endl;
    myTree.printInOrder();

    cout << "postfix: " << endl;
    myTree.printPostOrder();
    return 0;
}