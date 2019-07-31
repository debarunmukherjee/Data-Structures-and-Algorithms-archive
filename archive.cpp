#include<bits/stdc++.h>

using namespace std;

#define ll long long int

/*
    Algorithm name : Seive of Eratosthenes
    
    Usage : Finding all the prime numbers less than or equal to a given limit.

    Constraints : Usually 10,000,000 (10^7) or so.

    Constraint Variable used : MAX

    Algorithm : 

        Step 1 : Initialize a boolean array, say `primes` of size MAX with `true` value.

        Step 2 : Intialize a counter variable `p` with 2. This will be used to loop through the array above.

        Step 3 : If `primes[p]` is false, it indicates that `p` is a prime number.
                 Starting from `p*p` visit all the elements in `prime` array in steps of `p` and mark them as `false`. So visit `p*p` , `p*(p+1)` , `p*(p+2)` , ... <= MAX in the `primes` array and update
                 the values to `false`.

        Step 4 : Increment `p` till MAX and repeat " Step 3 ".
    
    Complexity : O( n*log(log(n)) )

    IDEone Link : https://ideone.com/6Eztjj
  
    Note : While using this function make sure to update the `MAX` variable
    as per the needs.
*/

bool * Sieve_of_eratosthenes()
{
    //Update the max limit as per needs
    const int MAX = 100;

    //Initialize the primes array with true initially
    bool * primes = new bool[MAX+1];

    //Initialize to true
    for (int i = 0; i < MAX+1; i++)
    {
        primes[i] = true;
    }

    //Loop through every value of primes array
    for(int p = 2; p <= MAX; p++)
    {
        if (primes[p])
        {
            for (int i = p*p; i <= MAX; i+=p)
            {
                primes[i]= false;
            }
        }
    }

    return primes;

    //Debugging
    // for(int i=0; i<= MAX; i++)
    //     if(primes[i])
    // 	    cout<<i<<"\n";
}

/*
    Algorithm name : Prime factorization - Extension of Sieve of Eratosthenes
    
    Usage : Finding all the distinct prime factors of a given number.

    Constraints : Usually used when querying a lot of numbers, with each query taking O( log n ) time.

    Constraint Variable used : `MAX` for the highest number till which the smallest prime factors are to be found
                               `n` for the number whose prime factors are to be found.

    Algorithm : 

        Step 1 : Get the smallest prime factors of all numbers according to the range to be used,
                 using the Sieve of Eratosthenes. Also, let the current number whose prime factors are to be found
                 be n

        Step 2 : Push the spf[n] into the ans.

        Step 3 : Update n as n = n/spf[n].

        Step 4 : Repeat Step 2 and Step 3 till n > 1.
    
    Complexity : O( n*log(log(n)) ) for finding the smallest prime factors of the numbers in the range to be used.
                 O( log n ) for finding the prime factors of a given number.

    IDEone Link : https://ideone.com/jHXYhB
  
    Note : 
        1)  While using this function make sure to update the `MAX` variable
            as per the needs.
        
        2)  Remove the `spf__global` variable from the `Get_spf` function, if not used.
*/

//Make the spf array global, so that it is available for every query without recalculation
int spf__global[101];

int * Get_spf()
{
    //Update the max limit as per needs
    const int MAX = 100;

    //Create the array to store the smallest prime factor
    int * spf = new int[MAX + 1];

    //Create the primes array with true initially
    bool * primes = new bool[MAX + 1];

    //Initialize the arrays
    for (int i = 0; i < MAX+1; i++)
    {
        primes[i] = true;
        spf[i] = 0;
        spf__global[i] = 0;
    }

    //Loop through every value of primes array
    for(int p = 2; p <= MAX; p++)
    {
        if (primes[p])
        {
            spf[p] = p;
            spf__global[p] = p;

            for (int i = p*p; i <= MAX; i+=p)
            {
                primes[i]= false;

                if(spf[i] == 0)
                {
                    spf[i] = p;
                    spf__global[i] = p;
                }
            }
        }
    }

    return spf;

    //Debugging
    // for(int i=0; i<= MAX; i++)
    // 	cout<<i<<" => "<<spf[i]<<"\n";
}

vector<int> prime_factorize_log(ll n)
{
    vector<int> prime_factors;

    while(n > 1)
    {
        prime_factors.push_back(spf__global[n]);
        n = n/spf__global[n];        
    }

    return prime_factors;
}