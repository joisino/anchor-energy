#pragma once

#include <iostream>
#include <queue>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>

using namespace std;

using ld = double;
using edge = pair<int, ld>;
using pi = pair<int, int>;

const ld INF = 1e18;

template<class T> ostream& operator << (ostream& os, const vector<T> v){
  for(int i = 0; i < v.size(); i++){if(i > 0){os << " ";} os << v[i];} return os;
}
template<class T> ostream& operator << (ostream& os, const vector<vector<T>> v){
  for(int i = 0; i < v.size(); i++){if(i > 0){os << endl;} os << v[i];} return os;
}

struct Unionfind{
  vector<int> size, par;
  Unionfind(){}
  Unionfind(int n) : size(n, 1), par(n){
    for(int i = 0; i < n; i++){
      par[i] = i;
    }
  }
  int find(int x){
    if(par[x] == x){
      return x;
    }
    return par[x] = find(par[x]);
  }
  bool unite(int x, int y){
    x = find(x);
    y = find(y);
    if(x == y){
      return false;
    }
    if(size[y] < size[x]){
      swap(x, y);
    }
    par[x] = y;
    size[y] += size[x];
    return true;
  }
  bool same(int x, int y){
    return find(x) == find(y);
  }
};

struct BIT{
  vector<ld> bit;
  int size;
  BIT(int n){
    size = n;
    bit = vector<ld>(n, 0);
  }
  void add(int k, ld v){
    for(int i = k + 1; i <= size; i += i & -i){
      bit[i-1] += v;
    }
  }
  ld sum(int k){
    ld res = 0;
    for(int i = k; i > 0; i -= i & -i){
      res += bit[i-1];
    }
    return res;
  }
};

struct Dijkstra{
  typedef pair<ld, int> pdi;  
  vector<vector<edge>> G;
  vector<ld> dist;
  int n;
  Dijkstra(vector<vector<edge>> arg_G) : G(arg_G){
    n = G.size();
  }
  vector<ld> dijkstra(int s){
    dist = vector<ld>(n, INF);
    dist[s] = 0;
    priority_queue<pdi,vector<pdi>,greater<pdi>> que;
    que.emplace(0, s);
    while(!que.empty()){
      ld d = que.top().first;
      int p = que.top().second;
      que.pop();
      if(d > dist[p]){
	continue;
      }
      for(edge w: G[p]){
	int to = w.first;
	ld co = w.second;
	if(d + co < dist[to]){
	  dist[to] = d + co;
	  que.emplace(dist[to], to);
	}
      }
    }
    return dist;
  }
};

struct SCC{
  int n, k;
  vector<vector<int>> G, rG;
  vector<int> vs, cmp;
  vector<bool> used;
  void init(int size){
    n = size;
    G = rG = vector<vector<int>>(n, vector<int>(0));
    cmp = vector<int>(n);
    vs.clear();
  }
  void add_edge(int a, int b){
    G[a].push_back(b);
    rG[b].push_back(a);
  }
  void dfs(int x){
    used[x] = true;
    for(int w: G[x]){
      if(!used[w]){
        dfs(w);
      }
    }
    vs.push_back(x);
  }
  void rdfs(int x){
    used[x] = true;
    cmp[x] = k;
    for(int w: rG[x]){
      if(!used[w]){
        rdfs(w);
      }
    }
  }
  int scc(){
    used = vector<bool>(n, false);
    for(int i = 0; i < n; i++){
      if(!used[i]){
        dfs(i);
      }
    }
    reverse(vs.begin(), vs.end());
    k = 0;
    used = vector<bool>(n, false);
    for(int w: vs){
      if(!used[w]){
        rdfs(w);
        k++;
      }
    }
    return k;
  }
};
