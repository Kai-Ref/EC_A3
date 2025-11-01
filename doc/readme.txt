How To Run:

Exercise 1:
    To run exercise 1, run the exercise1.py script via a command window (bash / powershell) and pass it with the following arguements
        - function name: either one (1+1), rls, or GA
        - the seed for the random number generator
        - fitness evalutation budget (most likely 10000 or 100000)
    An example of this would be: "python .\exercise1.py rls 500 10000". The results will be in Exercise_1/data/run

Exercise 2:
    To run exercise 2, simply execute the gsemo.py file without additional arguments. Careful: It is parallelised, using all available processes -> to be able to do other stuff while it runs the size of the pool might needs to be reduced.
