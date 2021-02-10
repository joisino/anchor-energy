#include <vector>
#include <algorithm>
#include <vector>
#include <set>
#include <string>
#include <map>
#include <cassert>
#include <random>
#include "util.hpp"

using namespace std;

extern mt19937 mt;

vector<vector<edge>> set_to_graph(int n, set<pi> ss){
  uniform_real_distribution<ld> unif;
  vector<vector<edge>> G(n, vector<edge>(0));
  for(pi e: ss){
    int a = e.first;
    int b = e.second;
    ld c = unif(mt);
    G[a].emplace_back(b, c);
    G[b].emplace_back(a, c);
  }
  return G;
}

set<pi> gen_tree_set(int n){
  Unionfind uf(n);
  set<pi> ss;
  while((int)ss.size() < n-1){
    int a = mt() % n;
    int b = mt() % n;
    if(uf.same(a, b)){
      continue;
    }
    uf.unite(a, b);
    ss.insert(pi(a, b));
  }
  return ss;
}

vector<vector<edge>> gen_tree(int n){
  return set_to_graph(n, gen_tree_set(n));
}

set<pi> gen_graph_set(int n, int m){
  set<pi> ss = gen_tree_set(n);
  while((int)ss.size() < m){
    int a = mt() % n;
    int b = mt() % n;
    if(a == b){
      continue;
    }
    if(a > b){
      swap(a, b);
    }
    ss.insert(pi(a, b));
  }
  return ss;
}

vector<vector<edge>> gen_graph(int n, int m){
  return set_to_graph(n, gen_graph_set(n, m));
}

vector<ld> gen_prob(int n){
  ld s = 0;
  vector<ld> res(n);
  for(int i = 0; i < n; i++){
    res[i] = mt();
    s += res[i];
  }
  for(int i = 0; i < n; i++){
    res[i] /= s;
  }
  return res;
}

vector<ld> gen_unif_prob(int n){
  ld s = 0;
  vector<ld> res(n);
  for(int i = 0; i < n; i++){
    res[i] = 1.0 / n;
  }
  return res;
}

vector<vector<ld>> graph_to_dist(vector<vector<edge>> G){
  int n = G.size();
  Dijkstra dij(G);
  vector<vector<ld>> d(n);
  vector<bool> valid(n, true);
  vector<int> idx(n, -1);
  int nn = 0;
  for(int i = 0; i < n; i++){
    d[i] = dij.dijkstra(i);
    int cnt = 0;
    for(int j = 0; j < n; j++){
      if(d[i][j] == INF){
        cnt++;
      }
    }
    if(cnt * 2 < n){
      idx[i] = nn++;
    }
  }
  vector<vector<ld>> res(nn, vector<ld>(nn));
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      if(idx[i] >= 0 && idx[j] >= 0){
        res[idx[i]][idx[j]] = d[i][j];
      }
    }
  }
  return res;
}

vector<vector<edge>> file_to_graph(string s, bool reindex=false, bool weighted=false, bool directed=false){
  FILE *fp = fopen(s.c_str(), "r");
  int n, m;
  assert(fscanf(fp, "%d %d", &n, &m) == 2);
  int cnt = 0;
  map<int, int> ma;
  auto idx = [&](int x){
    if(ma.find(x) == ma.end()){
      ma[x] = cnt++;
    }
    return ma[x];
  };
  vector<vector<edge>> G(n, vector<edge>(0));
  for(int i = 0; i < m; i++){
    int a, b;
    ld c = 1;
    assert(fscanf(fp, "%d %d", &a, &b) == 2);
    if(weighted){
      assert(fscanf(fp, "%lf", &c) == 1);
    }
    if(reindex){
      a = idx(a);
      b = idx(b);
    }
    G[a].emplace_back(b, c);
    if(!directed){
      G[b].emplace_back(a, c);
    }
  }
  return G;
}
