#include <iostream>
#include <vector>
#include <utility>
#include <cassert>
#include <random>
#include "util.hpp"

using namespace std;

vector<vector<edge>> load(string s){
  FILE *fp = fopen(s.c_str(), "r");
  int n, m;
  fscanf(fp, "%d %d", &n, &m);
  vector<vector<edge>> G(n, vector<edge>(0));
  for(int i = 0; i < m; i++){
    int a, b;
    fscanf(fp, "%d %d", &a, &b);
    G[a].emplace_back(b, 1);
    G[b].emplace_back(a, 1);
  }
  return G;
}

mt19937 mt(1234);

vector<ld> gen_unif_prob(int n);
vector<vector<ld>> graph_to_dist(vector<vector<edge>> G);

pair<ld, vector<vector<ld>>> anchor_energy_naive(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2);
pair<ld, vector<vector<ld>>> anchor_wasserstein(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2, ld eps=1);
pair<ld, vector<vector<ld>>> gromov_sinkhorn(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2, ld alpha=1, ld eta=0.1);

int main(int argc, char **argv){

  string type = argv[1];
  if(type == "AE"){
    assert(argc == 4);
  } else if(type == "AW"){
    assert(argc == 5); 
  } else if(type == "GW"){
    assert(argc == 6);
  }
  auto G1 = load(argv[argc-1]);
  auto d1 = graph_to_dist(G1);
  vector<ld> a1 = gen_unif_prob(G1.size());
  auto G2 = load(argv[argc-2]);
  auto d2 = graph_to_dist(G2);
  vector<ld> a2 = gen_unif_prob(G2.size());
  vector<vector<ld>> P;
  if(type == "AE"){
    P = anchor_energy_naive(d1, a1, d2, a2).second;
  } else if(type == "AW"){
    P = anchor_wasserstein(d1, a1, d2, a2, atof(argv[2])).second;
  } else if(type == "GW"){
    P = gromov_sinkhorn(d1, a1, d2, a2, atof(argv[2]), atof(argv[3])).second;
  }
  for(int i = 0; i < G1.size(); i++){
    ld ma = -1e9;
    vector<int> pos(0);
    for(int j = 0; j < G2.size(); j++){
      if(abs(ma - P[i][j]) < 1e-6){
        pos.push_back(j);
      } else if(ma < P[i][j]){
        ma = P[i][j];
        pos.clear();
        pos.push_back(j);
      }
    }
    int ans = pos[mt() % (int)(pos.size())];
    cout << ans << " ";
 }
  cout << endl;

  return 0;
}
