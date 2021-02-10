#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <cassert>
#include <chrono>
#include <random>
#include "util.hpp"

using namespace std;

vector<ld> gen_unif_prob(int n);

ld anchor_energy(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2);
ld robust_anchor_energy(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2);
ld anchor_energy_naive_value(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2);
pair<ld, vector<vector<ld>>> anchor_wasserstein(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2, ld eps=1);
pair<ld, vector<vector<ld>>> robust_anchor_wasserstein(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2, ld eps=1);
pair<ld, vector<vector<ld>>> gromov_sinkhorn(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2, ld alpha=1, ld eta=0.1);

vector<vector<ld>> compress(vector<vector<ld>> d);

mt19937 mt(1234);

pair<vector<vector<ld>>, vector<ld>> load(string filename, bool prob){
  ifstream ifs(filename);
  int n;
  ifs >> n;
  vector<ld> a = gen_unif_prob(n);
  if(prob){
    for(int i = 0; i < n; i++){
      ifs >> a[i];
    }
  }
  vector<vector<ld>> d(n, vector<ld>(n));
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      ifs >> d[i][j];
    }
  }
  return make_pair(d, a);
}

int main(int argc, char **argv){
  assert(argc >= 4);

  string type = argv[1];
  if(type == "AE" || type == "RAE" || type == "NAE"){
    assert(argc == 4);
  } else if(type == "AW" || type == "RAW"){
    assert(argc == 5); 
  } else if(type == "GW"){
    assert(argc == 6);
  }
  auto S1 = load(argv[argc-2], false);
  auto S2 = load(argv[argc-1], false);
  auto start = chrono::system_clock::now();
  ld value;
  if(type == "AE"){
    value = anchor_energy(S1.first, S1.second, S2.first, S2.second);
  } else if(type == "RAE"){
    value = robust_anchor_energy(S1.first, S1.second, S2.first, S2.second);
  } else if(type == "NAE"){
    value = anchor_energy_naive_value(S1.first, S1.second, S2.first, S2.second);
  } else if(type == "AW"){
    value = anchor_wasserstein(S1.first, S1.second, S2.first, S2.second, atof(argv[2])).first;
  } else if(type == "RAW"){
    value = robust_anchor_wasserstein(S1.first, S1.second, S2.first, S2.second, atof(argv[2])).first;
  } else if(type == "GW"){
    value = gromov_sinkhorn(S1.first, S1.second, S2.first, S2.second, atof(argv[2]), atof(argv[3])).first;
  } else if(type == "RGW"){
    value = gromov_sinkhorn(compress(S1.first), S1.second, compress(S2.first), S2.second, atof(argv[2]), atof(argv[3])).first;
  }
  auto end = chrono::system_clock::now();
  ld elapsed = static_cast<ld>(chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000);
  printf("%.6lf %.6lf\n", (double)value, (double)elapsed);

  return 0;
}
