#include<bits/stdc++.h>
#define loop(i,a,b) for(int i=a;i<b;i++)
#define loopb(i,a,b) for(int i=a;i>=b;i--)
#define loopm(i,a,b,step) for(int i=a;i<b;i+=step)
#define loopbm(i,a,b,step) for(int i=a;i>=b;i-=step)
#define pb push_back
#define mp(a,b) make_pair(a,b)
#define init(arr,val) memset(arr,val,sizeof(arr))
#define INF 1000000007
#define MOD 1000000007
#define BINF 1000000000000000001
#define int long long int
#define double long double
#define vi vector<int>
#define vitr vector<int>::iterator


const int P=101;
const int MOD2=1e9+7;
const int P2=103;

using namespace std;

const int N=1e5+2;
vector<pair<int,int> >graph[N]; // edge-weight

int readInt () {
	bool minus = false;
	int result = 0;
	char ch;
	ch = getchar();
	while (true) {
		if (ch == '-') break;
		if (ch >= '0' && ch <= '9') break;
		ch = getchar();
	}
	if (ch == '-') minus = true; else result = ch-'0';
	while (true) {
		ch = getchar();
		if (ch < '0' || ch > '9') break;
		result = result*10 + (ch - '0');
	}
	if (minus)
		return -result;
	else
		return result;
}


/*Maths*/
std::vector<long long> sieve(long long n)
{
	long long i,j,k;
	vector<bool> nos;
	nos.pb(false);
	nos.pb(false);
	for(i=1;i<n;i++)
		nos.pb(true);

	for(i=1; i*i<=n; i++)
	{
		if(nos[i])
		{
			for(j=2*i; j<=n; j+=i)
			{
				nos[j]=false;
			}
		}
	}

	vector<long long> primes;
	for(i=0; i<n; i++)
	{
		if(nos[i])
		{
			primes.pb(i);
		}
	}

	return primes;
}

//Multiply with Mod
long long mulmod(long long a, long long b, long long mod)
{
	long long x=0, y=a%mod;
	while(b>0)
	{
		if(b%2==1)
			x = (x+y)%mod;
		y= (y*2)%mod;
		b/=2;
	}
	return x%mod;
}
//Fast expo
long long power(long long x, long long y, long long mod)
{
	long long result= 1;
	x%= mod;
	while(y>0)
	{
		if(y%2==1)
		{
			result= mulmod(result,x,mod);//(result*x)%mod;
		}
		y/= 2;
		x= mulmod(x,x,mod);//(x*x)%mod;
	}
	return result;
}
//Inverse x^-1 % mod
long long inverse(long long x, long long mod)
{
	return power(x, mod-2, mod);
}

//Factorial pre calculate till n
std::vector<long long> getFactorial(long long n, long long mod)
{
	std::vector<long long> facts;
	facts.pb(1);
	for(long long i = 1; i<=n; i++)
	{
		facts.push_back( mulmod(facts[i-1], i, mod));
	}
	return facts;
}
// Direct nCr
long long nCr(long long n, long long r, long long mod)
{
	long long num=1, den=1, i=1;
	if(n==0 || n==1 || r==0)
		return 1;
	r= min(r, n-r);

	for(i=n-r+1; i<=n; i++)
		num=mulmod(num, i, mod);
	for(i=2; i<=r; i++)
		den=mulmod(den, i, mod);
	den= inverse(den, mod);
	//cout<<n SP<<r SP<<num SP<<den NL;
	return mulmod(num, den, mod);
}
//Fast nCr, with facts precalc
long long nCr(long long n, long long r, long long mod, std::vector<long long> facts)
{
	long long num= facts[n];
	long long den= mulmod(facts[r], facts[n-r], mod);
	den= inverse(den, mod);

	return mulmod(num, den, mod);
}

//Miller Rabin Test
bool millerTest(long long d, long long n)
{
	long long a = 2+ rand()%(n-4);
	long long x = power(a, d, n);

	if(x==1 || x== n-1)
		return true;
	while(d!= n-1)
	{
		x= mulmod(x,x,n);//(x*x)%n;
		d*=2;

		if(x==1)
			return false;
		if(x== n-1)
			return true;
	}
	return false;
}
bool isMillerPrime(long long n, long long k)
{
	if(n<=1 || n==4)
		return false;
	if(n<=3)
		return true;

	long long d= n-1;
	while(d%2==0)
		d/=2;

	for(long long i=0; i<k; i++)
		if( millerTest(d, n) == false)
			return false;
	return true;
}

//Factor Count
long long factorCount(long long N, std::vector<long long> primes)
{
	long long count=1;
	for(int i=0; i<primes.size() && N>1 ; i++)
	{
		if(primes[i] > (long long)cbrt(N)+1)
			break;
		long long expo= 1;
		while(N%primes[i]==0)
		{
			N/=primes[i];
			expo+= 1;
		}
		count=count*expo;
	}

	if(N==1)
		return count*1;

	if(isMillerPrime(N,10))
		return count*2;

	if(((long long)sqrt(N))*((long long)sqrt(N))== N )
		return count*3;
	else
		return count*4;
}

//Merge Sort + Inversion Count
std::vector<long long> arr; 
long long merge(long long l, long long mid, long long r)
{
	long long ans=0;
	std::vector<long long> A, B;

	for(int i=l; i<mid+1; i++)
		A.push_back(arr[i]);

	for(int i=mid+1; i<r+1; i++)
		B.push_back(arr[i]);

	long long ii=0, jj=0, kk=l;
	long long a=mid-l+1, b=r-mid;
 
	while(ii<a && jj<b)
	{
		if(A[ii] <= B[jj])
		{
			arr[kk]=A[ii];
			ii++;
		}
		else
		{
			arr[kk]=B[jj];
			jj++;
			ans+=a-ii;
		}
		kk++;
	}
	while(ii<a)
	{
		arr[kk]=A[ii];
		ii++;
		kk++;
	}
	while(jj<b)
	{
		arr[kk]=B[jj];
		jj++;
		kk++;
	}
 
	return ans;
}
 
long long mergeSort(long long l, long long r)
{
	if(l>=r)
		return 0;
	long long mid=l+(r-l)/2;
	long long ans=0;
	ans+=mergeSort(l, mid);
	ans+=mergeSort(mid+1, r);
	ans+=merge(l, mid, r);
	return ans;
}

/*DS*/

//Range Sum query
/*make array 0-based indexing*/
class RSUM
{
    vector<int>tree;
    public:
        RSUM(int n)
        {
            int size=4*n;
            loop(i,0,size)
             tree.pb(0);
        }

        void construct(int arr[],int n)
        {
            //arr[] is 0 based
            loop(i,0,n)
            tree[i+n]=arr[i];
            loopb(i,n-1,1)
             tree[i]=tree[i<<1]+tree[i<<1|1];
        }
        //point update
        void update(int idx,int val,int n)
        {
            // set arr[p]=val in main()
            idx+=n;
            tree[idx]=val;
            for(;idx>1;idx>>=1)
             tree[idx>>1]=tree[idx]+tree[idx^1];

        }
        //query from [l,r)
        int query(int l,int r,int n)
        {
            int res=0;
            for(l+=n,r+=n;l<r;l>>=1,r>>=1)
            {
                if(l&1) res+=tree[l++];
                if(r&1) res+=tree[--r];
            }
            return res;
        }
    ~RSUM()
    {
        tree.clear();
    }
};

//Range Max Query
class RMQ
{
     vector<int>tree;
    public:
        RMQ(int n)
        {
            int size=4*n;
            loop(i,0,size)
             tree.pb(0);
        }

        void construct(int arr[],int n)
        {
            //arr[] is 0 based
            loop(i,0,n)
            tree[i+n]=arr[i];
            loopb(i,n-1,1)
             tree[i]=max(tree[i<<1],tree[i<<1|1]);
        }
        //point update
        void update(int idx,int val,int n)
        {
            // set arr[p]=val in main()
            idx+=n;
            tree[idx]=val;
            for(;idx>1;idx>>=1)
             tree[idx>>1]=max(tree[idx],tree[idx^1]);

        }
        //query from [l,r)
        int query(int l,int r,int n)
        {
            int res=0;
            //note: In case of Range Min query or -ve values allowed set res accordingly
            for(l+=n,r+=n;l<r;l>>=1,r>>=1)
            {
                if(l&1) res=max(res,tree[l++]);
                if(r&1) res=max(res,tree[--r]);
            }
            return res;
        }
    ~RMQ()
    {
        tree.clear();
    }
};

class Lazy
{
  vector<int>tree;
  vector<int>lazy;
  public:
        Lazy(int n)
        {
            int size=4*n;
            loop(i,0,size)
            {
                tree.pb(0);
                lazy.pb(0);
            }

        }

         void construct(int arr[],int n)
        {
            //arr[] is 0 based
            loop(i,0,n)
            tree[i+n]=arr[i];
            loopb(i,n-1,1)
             tree[i]=tree[i<<1]+tree[i<<1|1];
        }

        void updateUtil(int beg,int end,int l,int r,int pos,int diff)
        {
            if(lazy[pos]!=0)
            {
                tree[pos]+=(end-beg+1)*lazy[pos];
                if(beg!=end)
                {
                    lazy[2*pos+1]+=lazy[pos];
                    lazy[2*pos+2]+=lazy[pos];
                }
                lazy[pos]=0;
            }
            if(l<=beg and r>=end)
            {
                tree[pos]+=(end-beg+1)*diff;
                if(beg!=end)
                {
                    lazy[2*pos+1]+=diff;
                    lazy[2*pos+2]+=diff;
                }
             return;
            }
            else if(l>end or r<beg or beg>end)
             return;

            else
            {
                int mid=(beg+end)/2;
                updateUtil(beg,mid,l,r,2*pos+1,diff);
                updateUtil(mid+1,end,l,r,2*pos+2,diff);
            }

        }

        //l,r is 0 based. Update [l,r]
        void update(int n,int l,int r,int diff)
        {
           updateUtil(0,n-1,l,r,0,diff);
        }

        int getans(int beg,int end,int l,int r,int pos)
        {
            if(lazy[pos]!=0)
            {
                tree[pos]+=(end-beg+1)*lazy[pos];
                if(beg!=end)
                {
                    lazy[2*pos+1]+=lazy[pos];
                    lazy[2*pos+2]+=lazy[pos];
                }
                lazy[pos]=0;
            }
            if(l<=beg and r>=end)
             return tree[pos];
            else if(l>end or r<beg or beg>end)
             return 0;
            else
            {
                int mid=(beg+end)/2;
                return getans(beg,mid,l,r,2*pos+1)+getans(mid+1,end,l,r,2*pos+2);
            }
        }

        //query from [l,r]
        int query(int n,int l,int r)
        {
            return getans(0,n-1,l,r,0);
        }

    ~Lazy()
    {
        tree.clear();
        lazy.clear();
    }

};


//Merge Sort Tree
class MSORTTree
{
    vector<vector<int> >tree;
    public:
        MSORTTree(int n)
        {
            int size=4*n;
            loop(i,0,size)
            {
                vector<int>temp;
                tree.pb(temp);
            }
        }

         void construct(int arr[],int n)
        {
            //arr[] is 0 based
            loop(i,0,n)
            tree[i+n].pb(arr[i]);
            loopb(i,n-1,1)
             {
                 merge(tree[i<<1].begin(),tree[i<<1].end(),tree[i<<1|1].begin(),tree[i<<1|1].end(),back_inserter(tree[i]));
             }
        }

       //write query function accordingly. example no.of elements >=k in range [l,r) is given,0-based indexing
       int countk(int n,int l,int r,int k)
       {
           int res=0;
           for(l+=n,r+=n;l<r;l>>=1,r>>=1)
           {
               if(l&1)
               {
                   vitr it=lower_bound(tree[l].begin(),tree[l].end(),k);
                   res+=tree[l].size()-(it-tree[l].begin());
                   l++;
               }
               if(r&1)
               {
                   r--;
                   vitr it=lower_bound(tree[r].begin(),tree[r].end(),k);
                   res+=tree[r].size()-(it-tree[r].begin());
               }
           }
           return res;
       }

};


class dijkstra
{
  int *d;
  bool *mark;
  public:
   dijkstra(int n)
   {
       d=new int[n+1];
       mark=new bool[n+1];
   }
   int getelement(int i)
   {
       return d[i];
   }
   void calcdistance(int v,int n)
   {
      loop(i,0,n+1)
      {
      	d[i]=BINF;
      	mark[i]=false;
      }
      d[v]=0;
      int u;
      set<pair<int,int> > s;
	s.insert({d[v], v});
	while(!s.empty()){
		u = s.begin() -> second;
		s.erase(s.begin());
		for(auto p : graph[u]) //adj[v][i] = pair(vertex, weight)
			if(d[p.first] > d[u] + p.second){
				s.erase({d[p.first], p.first});
				d[p.first] = d[u] + p.second;
				s.insert({d[p.first], p.first});
			}
	}
   }
};

struct wavelet_tree{
	int lo, hi;
	wavelet_tree *l, *r;
	vi b;
	
	//nos are in range [x,y]
	//array indices are [from, to)
	wavelet_tree(int *from, int *to, int x, int y){
		lo = x, hi = y;
		if(lo == hi or from >= to) return;
		int mid = (lo+hi)/2;
		auto f = [mid](int x){
			return x <= mid;
		};
		b.reserve(to-from+1);
		b.pb(0);
		for(auto it = from; it != to; it++)
			b.pb(b.back() + f(*it));
		//see how lambda function is used here	
		auto pivot = stable_partition(from, to, f);
		l = new wavelet_tree(from, pivot, lo, mid);
		r = new wavelet_tree(pivot, to, mid+1, hi);
	}
	
	//kth smallest element in [l, r]
	int kth(int l, int r, int k){
		if(l > r) return 0;
		if(lo == hi) return lo;
		int inLeft = b[r] - b[l-1];
		int lb = b[l-1]; //amt of nos in first (l-1) nos that go in left 
		int rb = b[r]; //amt of nos in first (r) nos that go in left
		if(k <= inLeft) return this->l->kth(lb+1, rb , k);
		return this->r->kth(l-lb, r-rb, k-inLeft);
	}
	
	//count of nos in [l, r] Less than or equal to k
	int LTE(int l, int r, int k) {
		if(l > r or k < lo) return 0;
		if(hi <= k) return r - l + 1;
		int lb = b[l-1], rb = b[r];
		return this->l->LTE(lb+1, rb, k) + this->r->LTE(l-lb, r-rb, k);
	}
  
	//count of nos in [l, r] equal to k
	int count(int l, int r, int k) {
		if(l > r or k < lo or k > hi) return 0;
		if(lo == hi) return r - l + 1;
		int lb = b[l-1], rb = b[r], mid = (lo+hi)/2;
		if(k <= mid) return this->l->count(lb+1, rb, k);
		return this->r->count(l-lb, r-rb, k);
	}
	~wavelet_tree(){
		delete l;
		delete r;
	}
};



/*prefix-hash*/

int mod[N],invmod[N];
int mod1[N],invmod1[N];


int fxp(int a,int b,int mod)
{
    if(b==0)
     return 1;
    else if(b==1)
     return a;
    else
    {
        if(b%2==0)
        {
            return ((fxp(a,b/2,mod)%mod)*(fxp(a,b/2,mod)%mod))%mod;
        }
        else
        {
            return (a*(fxp(a,b/2,mod)%mod)*(fxp(a,b/2,mod)%mod))%mod;
        }
    }
}

void EE(int a,int b,int &x,int &y)
{
    if(a%b==0)
    {
        x=0;
        y=1;
        return;
    }

    EE(b,a%b,x,y);
    int temp=x;
    x=y;
    y=temp-y*(a/b);
}

int inverse1(int a,int m)
{
    int x,y;
    EE(a,m,x,y);
    if(x<0)
     x+=m;
    return x;
}

int prefix1[N],prefix2[N];

int prefix_hash1(int l,int r)
{
    int ans;
    if(l==0)
     ans=prefix1[r];

    else
    {
        ans=prefix1[r]-prefix1[l-1];
        if(ans<MOD)
          ans+=MOD;

        ans*=invmod[l];
        if(ans>=MOD)
          ans%=MOD;
    }
    return ans;
}

int prefix_hash2(int l,int r)
{
    int ans;
    if(l==0)
     ans=prefix2[r];

    else
    {
        ans=prefix2[r]-prefix2[l-1];
        if(ans<MOD2)
          ans+=MOD2;

        ans*=invmod1[l];
        if(ans>=MOD2)
          ans%=MOD2;
    }
    return ans;
}


void calchash(string s)
{
    init(mod,0);
    init(invmod,0);
    init(mod1,0);
    init(invmod1,0);
    int n=s.size();
    mod[0]=1;
    invmod[0]=1;
    mod1[0]=1;
    invmod1[0]=1;
    int tmp=inverse1(P,MOD);
    int tmp1=inverse1(P2,MOD2);
    loop(i,1,N)
    {
        mod[i]=P*mod[i-1];
        invmod[i]=tmp*invmod[i-1];
        if(mod[i]>=MOD)
          mod[i]%=MOD;
        if(invmod[i]>=MOD)
          invmod[i]%=MOD;
    }

    loop(i,1,N)
    {
        mod1[i]=P2*mod1[i-1];
        invmod1[i]=tmp1*invmod1[i-1];
        if(mod1[i]>=MOD2)
         mod1[i]%=MOD2;
         if(invmod1[i]>=MOD2)
          invmod1[i]%=MOD2;
    }
   
   int cur=0;
      loop(i,0,n)
      {
          int temp=(int)(s[i]-'a')*mod[i];
          if(temp>=MOD)
           temp%=MOD;
          cur+=temp;
          if(cur>=MOD)
           cur%=MOD;

           prefix1[i]=cur;
      }
      cur=0;

        loop(i,0,n)
      {
          int temp=(int)(s[i]-'a')*mod1[i];
          if(temp>=MOD2)
           temp%=MOD2;
          cur+=temp;
          if(cur>=MOD2)
           cur%=MOD2;

           prefix2[i]=cur;
      }

}


//disjoint set union

class dsu
{
	vi arr;
	vi subtree;
	int n;
	public:
		dsu(int sz)
		{
			n=sz;
			loop(i,0,n+1){
			 arr.pb(i);
			 subtree.pb(1);
			}
		}
		
		int root(int x)
		{
			while(arr[x]!=x)
			{
				arr[x]=arr[arr[x]];//Path Compression
				x=arr[x];
			}
			return x;
		}
		
		void weighted_union(int a,int b)
		{
			int root1=root(a);
			int root2=root(b);
			if(subtree[root1]<subtree[root2])
			{
				arr[root1]=root2;
				subtree[root2]+=subtree[root1];
			}
			else
			{
				arr[root2]=root1;
				subtree[root1]+=subtree[root2];
			}
		}
	  bool find(int a,int b)
	  {
	  	 int root1=root(a);
	  	 int root2=root(b);
	  	 return (root1==root2?true:false);
	  }
};

/*-----------------------------TOPO SORT-----------------------*/
typedef double TYPE;
typedef vector<TYPE> VT;
typedef vector<VT> VVT;

typedef vector<int> VI;
typedef vector<VI> VVI;

bool TopologicalSort (const VVI &w, VI &order){
  int n = w.size();
  VI parents (n);
  queue<int> q;
  order.clear();

  for (int i = 0; i < n; i++){
    for (int j = 0; j < n; j++)
      if (w[j][i]) parents[i]++;
    if (parents[i] == 0) q.push (i);
  }

  while (q.size() > 0){
    int i = q.front();
    q.pop();
    order.push_back (i);
    for (int j = 0; j < n; j++) if (w[i][j]){
      parents[j]--;
      if (parents[j] == 0) q.push (j);
    }
  }

  return (order.size() == n);
}



/*-------------------------------------------------------PERSISTENT SEGMENT TREE---------------------------------*/
int l[SIZE],r[SIZE],st[SIZE],NODES=0;
int newleaf(int value)
{
    int p=++NODES;
    l[p]=r[p]=0;
    st[p]=value;
    return p;
}
int newparent(int lef,int rig)
{
    int p=++NODES;
    l[p]=lef,r[p]=rig;
    st[p]=st[lef]+st[rig];
    return p;
}
int build(int arr[],int L=0,int R=n-1)
{
    if(L==R) return newleaf(arr[L]);
    else return newparent(build(arr[],L,M),build(arr[],M+1,R));
}
//usage: int root=build(arr,0,n-1);
int update(int i,int x,int p,int L=0,int R=n-1)
{
    if(L==R) return newleaf(st[p]+x);
    if(i<=M) return newparent(update(i,x,l[p],L,M),r[p]);
    else return newparent(l[p],update(i,x,M+1,r[p]));
}
//usage:int newroot=update(i,x,root);

int rangecopy(int a,int b,int p,int revert,int L=0,int R=n-1)
{
    if(b<L or R<a) return p;//keep version
    if(a<=L and R<=b) return revert; //reverted version
    return newparent(rangecopy(a,b,l[p],l[revert],L,M),rangecopy(a,b,r[p],r[revert],M+1,R));
}
//usage:int revertedroot=revert(a,b,root,old_version_root)
/*---------------------------------------------------------------CONVEX HULL TRICK-------------------------------------*/
int pointer; //Keeps track of the best line from previous query
vector<long long> M; //Holds the slopes of the lines in the envelope
vector<long long> B; //Holds the y-intercepts of the lines in the envelope
//Returns true if either line l1 or line l3 is always better than line l2
bool bad(int l1,int l2,int l3)
{
	/*
	intersection(l1,l2) has x-coordinate (b1-b2)/(m2-m1)
	intersection(l1,l3) has x-coordinate (b1-b3)/(m3-m1)
	set the former greater than the latter, and cross-multiply to
	eliminate division
	*/
	return (B[l3]-B[l1])*(M[l1]-M[l2])<(B[l2]-B[l1])*(M[l1]-M[l3]);
}
//Adds a new line (with lowest slope) to the structure
void add(long long m,long long b)
{
	//First, let's add it to the end
	M.push_back(m);
	B.push_back(b);
	//If the penultimate is now made irrelevant between the antepenultimate
	//and the ultimate, remove it. Repeat as many times as necessary
	while (M.size()>=3&&bad(M.size()-3,M.size()-2,M.size()-1))
	{
		M.erase(M.end()-2);
		B.erase(B.end()-2);
	}
}
//Returns the minimum y-coordinate of any intersection between a given vertical
//line and the lower envelope
long long query(long long x)
{
	//If we removed what was the best line for the previous query, then the
	//newly inserted line is now the best for that query
	if (pointer>=M.size())
		pointer=M.size()-1;
	//Any better line must be to the right, since query values are
	//non-decreasing
	while (pointer<M.size()-1&&
	  M[pointer+1]*x+B[pointer+1]<M[pointer]*x+B[pointer])
		pointer++;
	return M[pointer]*x+B[pointer];
}
int main()
{
	int M,N,i;
	pair<int,int> a[50000];
	pair<int,int> rect[50000];
	freopen("acquire.in","r",stdin);
	freopen("acquire.out","w",stdout);
	scanf("%d",&M);
	for (i=0; i<M; i++)
		scanf("%d %d",&a[i].first,&a[i].second);
	//Sort first by height and then by width (arbitrary labels)
	sort(a,a+M);
	for (i=0,N=0; i<M; i++)
	{
		/*
		When we add a higher rectangle, any rectangles that are also
		equally thin or thinner become irrelevant, as they are
		completely contained within the higher one; remove as many
		as necessary
		*/
		while (N>0&&rect[N-1].second<=a[i].second)
			N--;
		rect[N++]=a[i]; //add the new rectangle
	}
	long long cost;
	add(rect[0].second,0);
	//initially, the best line could be any of the lines in the envelope,
	//that is, any line with index 0 or greater, so set pointer=0
	pointer=0;
	for (i=0; i<N; i++) //discussed in article
	{
		cost=query(rect[i].first);
		if (i < N-1)
			add(rect[i+1].second,cost);
	}
	printf("%lld\n",cost);
	return 0;
}

/*----------------------------------------------------------CENTROID DECOMPOSITION-----------------------------------------*/
/*----------- Pre-Processing ------------*/
void dfs0(int u)
{
	for(auto it=g[u].begin();it!=g[u].end();it++)
		if(*it!=DP[0][u])
		{
			DP[0][*it]=u;
			level[*it]=level[u]+1;
			dfs0(*it);
		}
}
void preprocess()
{
	level[0]=0;
	DP[0][0]=0;
	dfs0(0);
	for(int i=1;i<LOGN;i++)
		for(int j=0;j<n;j++)
			DP[i][j] = DP[i-1][DP[i-1][j]];
}
int lca(int a,int b)
{
	if(level[a]>level[b])swap(a,b);
	int d = level[b]-level[a];
	for(int i=0;i<LOGN;i++)
		if(d&(1<<i))
			b=DP[i][b];
	if(a==b)return a;
	for(int i=LOGN-1;i>=0;i--)
		if(DP[i][a]!=DP[i][b])
			a=DP[i][a],b=DP[i][b];
	return DP[0][a];
}
int dist(int u,int v)
{
	return level[u] + level[v] - 2*level[lca(u,v)];
}
/*-----------------Decomposition Part--------------------------*/
int nn;
void dfs1(int u,int p)
{
	sub[u]=1;
	nn++;
	for(auto it=g[u].begin();it!=g[u].end();it++)
		if(*it!=p)
		{
			dfs1(*it,u);
			sub[u]+=sub[*it];
		}
}
int dfs2(int u,int p)
{
	for(auto it=g[u].begin();it!=g[u].end();it++)
		if(*it!=p && sub[*it]>nn/2)
			return dfs2(*it,u);
	return u;
}
void decompose(int root,int p)
{
	nn=0;
	dfs1(root,root);
	int centroid = dfs2(root,root);
	if(p==-1)p=centroid;
	par[centroid]=p;
	for(auto it=g[centroid].begin();it!=g[centroid].end();it++)
	{
		g[*it].erase(centroid);
		decompose(*it,centroid);
	}
	g[centroid].clear();
}
/*----------------- Handle the Queries -----------------*/
void update(int u)
{
	int x = u;
	while(1)
	{
		ans[x] = min(ans[x],dist(x,u));
		if(x==par[x])
			break;
		x = par[x];
	}
}
int query(int u)
{
	int x = u;
	int ret=INF;
	while(1)
	{
		ret = min(ret,dist(u,x) + ans[x]);
		if(x==par[x])
			break;
		x = par[x];
	}
	return ret;
}
/*------------------------------------------LONGEST INCREASING SUBSEQUENCE----------------*/
int CeilIndex(std::vector<int> &v, int l, int r, int key) {
    while (r-l > 1) {
    int m = l + (r-l)/2;
    if (v[m] >= key)
        r = m;
    else
        l = m;
    }
 
    return r;
}
 
int LongestIncreasingSubsequenceLength(std::vector<int> &v) {
    if (v.size() == 0)
        return 0;
 
    std::vector<int> tail(v.size(), 0);
    int length = 1; // always points empty slot in tail
 
    tail[0] = v[0];
    for (size_t i = 1; i < v.size(); i++) {
        if (v[i] < tail[0])
            // new smallest value
            tail[0] = v[i];
        else if (v[i] > tail[length-1])
            // v[i] extends largest subsequence
            tail[length++] = v[i];
        else
            // v[i] will become end candidate of an existing subsequence or
            // Throw away larger elements in all LIS, to make room for upcoming grater elements than v[i]
            // (and also, v[i] would have already appeared in one of LIS, identify the location and replace it)
            tail[CeilIndex(tail, -1, length-1, v[i])] = v[i];
    }
 
    return length;
}
/*------------------------------Z ALGO---------------------------------------------------*/
int L = 0, R = 0;
for (int i = 1; i < n; i++) {
  if (i > R) {
    L = R = i;
    while (R < n && s[R-L] == s[R]) R++;
    z[i] = R-L; R--;
  } else {
    int k = i-L;
    if (z[k] < R-i+1) z[i] = z[k];
    else {
      L = i;
      while (R < n && s[R-L] == s[R]) R++;
      z[i] = R-L; R--;
    }
  }
}

/*-------------------------------------------FIBONACII IN LOGN---------------------------------*/
const long M = 1000000007; // modulo
map<long, long> F;

long f(long n) {
	if (F.count(n)) return F[n];
	long k=n/2;
	if (n%2==0) { // n=2*k
		return F[n] = (f(k)*f(k) + f(k-1)*f(k-1)) % M;
	} else { // n=2*k+1
		return F[n] = (f(k)*f(k+1) + f(k-1)*f(k)) % M;
	}
}

main(){
	long n;
	F[0]=F[1]=1;
	while (cin >> n)
	cout << (n==0 ? 0 : f(n-1)) << endl;
}

/*--------------------------------------------------------------------------BIPARTITE MATCHING--------------------------------------------------------*/
// This code performs maximum bipartite matching.
// It has a heuristic that will give excellent performance on complete graphs
// where rows <= columns.
//
//   INPUT: w[i][j] = cost from row node i and column node j or NO_EDGE
//   OUTPUT: mr[i] = assignment for row node i or -1 if unassigned
//           mc[j] = assignment for column node j or -1 if unassigned
//
//   BipartiteMatching returns the number of matches made.
//
// Contributed by Andy Lutomirski.

typedef vector<int> VI;
typedef vector<VI> VVI;

const int NO_EDGE = -(1<<30);  // Or any other value.

bool FindMatch(int i, const VVI &w, VI &mr, VI &mc, VI &seen)
{
  if (seen[i])
    return false;
  seen[i] = true;
  for (int j = 0; j < w[i].size(); j++) {
    if (w[i][j] != NO_EDGE && mc[j] < 0) {
      mr[i] = j;
      mc[j] = i;
      return true;
    }
  }
  for (int j = 0; j < w[i].size(); j++) {
    if (w[i][j] != NO_EDGE && mr[i] != j) {
      if (mc[j] < 0 || FindMatch(mc[j], w, mr, mc, seen)) {
	mr[i] = j;
	mc[j] = i;
	return true;
      }
    }
  }
  return false;
}

int BipartiteMatching(const VVI &w, VI &mr, VI &mc)
{
  mr = VI (w.size(), -1);
  mc = VI(w[0].size(), -1);
  VI seen(w.size());

  int ct = 0;
  for(int i = 0; i < w.size(); i++)
    {
      fill(seen.begin(), seen.end(), 0);
      if (FindMatch(i, w, mr, mc, seen)) ct++;
    }
  return ct;
}

/*--------------------------HLD----------------------------------*/
#include<bits/stdc++.h>
using namespace std;

typedef vector<int> vi;
typedef pair<int,int> pii;
typedef long long int lld;

#define sz                           size()
#define pb                           push_back
#define mp                           make_pair
#define F                            first
#define S                            second
#define fill(a,v)                    memset((a),(v),sizeof (a))
#define INF                          INT_MAX
#define mod 1000000007
#define __sync__		     std::ios::sync_with_stdio(false);

const int N = 10100;

class segTree
{
	public:
		struct tree
		{
			int val;
		};
		int n;
		vector< tree > st;
		segTree(int _n)
		{
			n = _n;
			st.resize(6*n + 10);
		}
		segTree() {}
		~segTree()
		{
			st.clear();
			n = 0;
		}

		tree merge(tree A,tree B)
		{
			tree C;
			C.val = max(A.val,B.val);
			return C;
		}

		void update(int s,int e,int node,int pos,int v)
		{
			if(s>e) return;
			if(s==e)
			{
				st[node].val = v;
				return;
			}

			int l = node<<1;
			int r = l|1;
			int m = (s+e)>>1;

			if(pos>m)
				update(m+1,e,r,pos,v);
			else
				update(s,m,l,pos,v);

			st[node] = merge(st[l],st[r]);

		}

		void update(int pos,int v)
		{
			update(0,n-1,1,pos,v);
		}

		tree query(int s,int e,int a,int b,int node)
		{
			if((s>=a && e<=b)) return st[node];
			tree L,R,A;
			int l = node<<1;
			int r = l|1;
			int m = (s+e)>>1;
			if(a>m) return query(m+1,e,a,b,r);
			if(b<=m) return query(s,m,a,b,l);

			R = query(m+1,e,a,b,r);
			L = query(s,m,a,b,l);
			return merge(R,L);
		}

		int query(int l, int r)
		{
			if(l>r) return 0;
			return query(0,n-1,l,r,1).val;
		}
};

int chainNo[N], chainHead[N], height[N], par[N];
int baseArr[N], posInBase[N];
int pos = 0;
int p[20][N];
int idx = 0;
int sub[N];
int LN = 14;
vector< pii > edge;
segTree st(N);
vi adj[N],cost[N];

int lca(int u,int v)
{
	if(height[u]>height[v]) swap(u,v);
	int diff = height[v] - height[u];
	for(int j=0;j<LN;j++) if((diff>>j)&1) v = p[j][v];
	if(u==v) return u;
	for(int j=LN-1;j>=0;j--)
		if(p[j][u]!=p[j][v])
		{
			u = p[j][u];
			v = p[j][v];
		}
	return p[0][v];
}


void dfs(int node, int h, int p = -1)
{
	sub[node] = 1;
	height[node] = h;
	par[node] = p;
	for(int i=0;i<adj[node].sz;i++)
		if(adj[node][i] != p)
		{
			dfs(adj[node][i], h+1, node);
			sub[node] += sub[adj[node][i]];
		}
}

void hld(int node,int cst,int p = -1)
{
	if(chainHead[idx]==-1)
		chainHead[idx] = node;

	chainNo[node] = idx;
	posInBase[node] = pos;
	baseArr[pos++] = cst;

	int ch = -1,mm = 0,cc = 0;
	for(int i=0;i<adj[node].sz;i++)
	{
		int u = adj[node][i];
		if(u!=p)
			if(sub[u]>mm)
			{
				ch = u;
				cc = cost[node][i];
				mm = sub[u];
			}
	}
	if(ch!=-1)
		hld(ch,cc,node);
	for(int i=0;i<adj[node].sz;i++)
	{
		int u = adj[node][i];
		if(u!=p && u!=ch)
		{
			idx++;
			hld(u,cost[node][i],node);
		}
	}
}

int main()
{
	//__sync__;
	int n,t;
	int xx,yy,zz;
	scanf("%d",&t);
	while(t--)
	{
		scanf("%d",&n);
		for(int i=0;i<n+2;i++)
		{
			adj[i].clear();
			cost[i].clear();
			edge.clear();
			chainHead[i]=-1;
		}
		fill(p,-1);
		pos = 0;
		idx = 0;
		for(int i=0;i<n-1;i++)
		{
			scanf("%d%d%d",&xx,&yy,&zz);
			xx--;yy--;
			adj[xx].pb(yy);
			adj[yy].pb(xx);
			cost[xx].pb(zz);
			cost[yy].pb(zz);
			edge.pb(mp(xx,yy));
		}
		dfs(0,0);
		hld(0,0);
		for(int i=0;i<pos;i++)
			st.update(i,baseArr[i]);
		for(int i=0;i<n;i++)
			p[0][i] = par[i];

		for(int j=1;j<LN;j++)
			for(int i=0;i<n;i++)
				if(p[j-1][i]!=-1)
					p[j][i] = p[j-1][p[j-1][i]];

		char s[100];
		int a,b;
		while(scanf("%s",s))
		{
			if(s[0]=='D') break;
			scanf("%d%d",&a,&b);
			if(s[0]=='C')
			{
				a--;
				int u = edge[a].F;
				int v = edge[a].S;
				if(par[u] == v) swap(u,v);
				st.update(posInBase[v],b);
			}
			else if(s[0]=='Q')
			{
				a--;b--;
				int u = a;
				int v = b;
				int l = lca(u,v);
				int ans = INT_MIN;
				while(chainNo[l]!=chainNo[u])
				{
					ans = max(ans,st.query(posInBase[chainHead[chainNo[u]]],posInBase[u]));
					u = chainHead[chainNo[u]];
					u = par[u];
				}
				ans = max(ans,st.query(posInBase[l]+1,posInBase[u]));
				u = v;
				while(chainNo[l]!=chainNo[u])
				{
					ans = max(ans,st.query(posInBase[chainHead[chainNo[u]]],posInBase[u]));
					u = chainHead[chainNo[u]];
					u = par[u];
				}
				ans = max(ans,st.query(posInBase[l]+1,posInBase[u]));
				printf("%d\n",ans);
			}
		}
	}
	return 0;
}



#undef int
int main()
{
#define int long long int
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
    

    
    return 0;
}