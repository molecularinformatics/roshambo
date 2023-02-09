int foo(void);


int foo(void)
{
    return 5;
}



int main(int argc, char **argv)
{
    int foo_val;
    foo_val = foo();
    return foo_val;
}
