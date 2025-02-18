#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#define SIZE 10

typedef struct Stack {
    int top;
    int* data;
}Stack;

bool Full(Stack* s) {
    return (s->top == SIZE - 1) ? true : false;
}

bool Empty(Stack* s) {
    return (s->top == -1) ? true : false;
}

void Push(Stack* s, int data) {
    if(Full(s)){
        //write later
        printf("Hello, world\n");
    }

    s->data[s->top++] = data;
}

void Pop(Stack* s) {
    if(Empty(s)) {
        printf("The stack is empty!\n");
        return;
    }

    s->top--;
}

void CreateStack(Stack* s) {
    s = malloc(sizeof(Stack));
    s->data = malloc(SIZE*sizeof(int));
    s->top = -1;
}

int main() {
    Stack stack;
    CreateStack(&stack);
}