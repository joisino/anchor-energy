#include <vector>
#include <algorithm>
#include <utility>
#include <tuple>
#include <numeric>
#include <cmath>
#include "util.hpp"

using namespace std;

vector<vector<ld>> log_sinkhorn(vector<vector<ld>> lK, vector<ld> a1, vector<ld> a2);

pair<ld, vector<vector<ld>>> anchor_energy_naive(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2){
  int n1 = a1.size();
  int n2 = a2.size();
  vector<vector<pair<ld, int>>> di1(n1);
  vector<vector<pair<ld, int>>> di2(n2);
  for(int i = 0; i < n1; i++){
    di1[i].clear();
    for(int k = 0; k < n1; k++){
      di1[i].emplace_back(d1[i][k], k);
    }
    sort(di1[i].begin(), di1[i].end());
    ld c = 0;
    for(int k = 0; k < n1; k++){
      di1[i][k].first = c;
      c += a1[di1[i][k].second];
    }
  }
  for(int j = 0; j < n2; j++){
    di2[j].clear();
    for(int k = 0; k < n2; k++){
      di2[j].emplace_back(d2[j][k], k);
    }
    sort(di2[j].begin(), di2[j].end());
    ld c = 0;
    for(int k = 0; k < n2; k++){
      di2[j][k].first = c;
      c += a2[di2[j][k].second];
    }
  }
  ld value = 0;
  vector<vector<ld>> P(n1, vector<ld>(n2, 0));
  for(int i = 0; i < n1; i++){
    for(int j = 0; j < n2; j++){
      ld coeff = a1[i] * a2[j];
      int k1 = 0;
      int k2 = 0;
      int it1 = 0;
      int it2 = 0;
      ld prv = 0;
      while(it1 < n1 || it2 < n2){
        if(it1 == n1 || (it2 < n2 && di1[i][it1].first > di2[j][it2].first)){
          value += coeff * abs(d1[i][k1] - d2[j][k2]) * (di2[j][it2].first - prv);
          P[k1][k2] += coeff * (di2[j][it2].first - prv);
          prv = di2[j][it2].first;
          k2 = di2[j][it2].second;
          it2++;
        } else {
          value += coeff * abs(d1[i][k1] - d2[j][k2]) * (di1[i][it1].first - prv);
          P[k1][k2] += coeff * (di1[i][it1].first - prv);
          prv = di1[i][it1].first;
          k1 = di1[i][it1].second;
          it1++;
        }
      }
      value += coeff * abs(d1[i][k1] - d2[j][k2]) * (1 - prv);
      P[k1][k2] += coeff * (1 - prv);
    }
  }
  return make_pair(value, P);
}

ld anchor_energy_naive_value(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2){
  int n1 = a1.size();
  int n2 = a2.size();
  vector<vector<pair<ld, int>>> di1(n1);
  vector<vector<pair<ld, int>>> di2(n2);
  for(int i = 0; i < n1; i++){
    di1[i].clear();
    for(int k = 0; k < n1; k++){
      di1[i].emplace_back(d1[i][k], k);
    }
    sort(di1[i].begin(), di1[i].end());
  }
  for(int j = 0; j < n2; j++){
    di2[j].clear();
    for(int k = 0; k < n2; k++){
      di2[j].emplace_back(d2[j][k], k);
    }
    sort(di2[j].begin(), di2[j].end());
  }
  ld value = 0;
  for(int i = 0; i < n1; i++){
    for(int j = 0; j < n2; j++){
      ld coeff = a1[i] * a2[j];
      ld c1 = 0;
      ld c2 = 0;
      int it1 = 0;
      int it2 = 0;
      ld prv = 0;
      while(it1 < n1 || it2 < n2){
        if(it1 == n1 || (it2 < n2 && di1[i][it1].first > di2[j][it2].first)){
          value += coeff * (di2[j][it2].first - prv) * abs(c1 - c2);
          c2 += a2[di2[j][it2].second];
          prv = di2[j][it2].first;
          it2++;
        } else {
          value += coeff * (di1[i][it1].first - prv) * abs(c1 - c2);
          c1 += a1[di1[i][it1].second];
          prv = di1[i][it1].first;
          it1++;
        }
      }
    }
  }
  return value;
}

ld anchor_energy(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2){
  int n1 = a1.size();
  int n2 = a2.size();
  vector<vector<pair<ld, int>>> di1(n1);
  vector<vector<pair<ld, int>>> di2(n2);
  vector<tuple<ld, int, int, int>> ds;
  for(int i = 0; i < n1; i++){
    di1[i].clear();
    for(int k = 0; k < n1; k++){
      di1[i].emplace_back(d1[i][k], k);
      ds.emplace_back(d1[i][k], 0, i, k);
    }
    sort(di1[i].begin(), di1[i].end());
  }
  for(int j = 0; j < n2; j++){
    di2[j].clear();
    for(int k = 0; k < n2; k++){
      di2[j].emplace_back(d2[j][k], k);
      ds.emplace_back(d2[j][k], 1, j, k);
    }
    sort(di2[j].begin(), di2[j].end());
  }
  sort(ds.begin(), ds.end());
  vector<ld> cs(0);
  cs.push_back(0);
  {
    vector<ld> c1(n1, 0);
    vector<ld> c2(n2, 0);
    for(int it = 0; it < (int)ds.size(); it++){
      ld dist;
      int i, j, k;
      tie(dist, i, j, k) = ds[it];
      if(i == 0){
        c1[j] += a1[k];
        cs.push_back(c1[j]);
      } else {
        c2[j] += a2[k];
        cs.push_back(c2[j]);
      }
    }
  }
  sort(cs.begin(), cs.end());
  cs.erase(unique(cs.begin(), cs.end()), cs.end());
  vector<int> yd_bef(ds.size());
  vector<int> yd_aft(ds.size());
  {
    vector<ld> c1(n1, 0);
    vector<ld> c2(n2, 0);
    for(int it = 0; it < (int)ds.size(); it++){
      ld dist;
      int i, j, k;
      tie(dist, i, j, k) = ds[it];
      if(i == 0){
        yd_bef[it] = lower_bound(cs.begin(), cs.end(), c1[j]) - cs.begin();
        c1[j] += a1[k];
        yd_aft[it] = lower_bound(cs.begin(), cs.end(), c1[j]) - cs.begin();
      } else {
        yd_bef[it] = lower_bound(cs.begin(), cs.end(), c2[j]) - cs.begin();
        c2[j] += a2[k];
        yd_aft[it] = lower_bound(cs.begin(), cs.end(), c2[j]) - cs.begin();
      }
    }
  }
  BIT t1((int)cs.size());
  BIT s1((int)cs.size());
  BIT t2((int)cs.size());
  BIT s2((int)cs.size());
  ld mass_sum1 = accumulate(a1.begin(), a1.end(), ld(0));
  ld mass_sum2 = accumulate(a2.begin(), a2.end(), ld(0));
  s1.add(0, mass_sum1);
  s2.add(0, mass_sum2);
  ld ans = 0;
  vector<ld> c1(n1, 0);
  vector<ld> c2(n2, 0);
  ld sc1 = 0;
  ld sc2 = 0;
  ld f = 0;
  ld prv = 0;
  for(int it = 0; it < (int)ds.size(); it++){
    ld dist;
    int type, j, k;
    tie(dist, type, j, k) = ds[it];
    ans += (dist - prv) * f;
    if(type == 0){
      s1.add(yd_bef[it], -a1[j]);
      t1.add(yd_bef[it], -a1[j] * c1[j]);
      ld s2s = s2.sum(yd_bef[it]);
      ld t2s = t2.sum(yd_bef[it]);
      f -= a1[j] * (s2s * c1[j] - t2s + (sc2 - t2s) - (mass_sum2 - s2s) * c1[j]);
      c1[j] += a1[k];
      sc1 += a1[k] * a1[j];
      s1.add(yd_aft[it], a1[j]);
      t1.add(yd_aft[it], a1[j] * c1[j]);
      s2s = s2.sum(yd_aft[it]);
      t2s = t2.sum(yd_aft[it]);
      f += a1[j] * (s2s * c1[j] - t2s + (sc2 - t2s) - (mass_sum2 - s2s) * c1[j]);
    } else {
      s2.add(yd_bef[it], -a2[j]);
      t2.add(yd_bef[it], -a2[j] * c2[j]);
      ld s1s = s1.sum(yd_bef[it]);
      ld t1s = t1.sum(yd_bef[it]);
      f -= a2[j] * (s1s * c2[j] - t1s + (sc1 - t1s) - (mass_sum1 - s1s) * c2[j]);
      c2[j] += a2[k];
      sc2 += a2[k] * a2[j];
      s2.add(yd_aft[it], a2[j]);
      t2.add(yd_aft[it], a2[j] * c2[j]);
      s1s = s1.sum(yd_aft[it]);
      t1s = t1.sum(yd_aft[it]);
      f += a2[j] * (s1s * c2[j] - t1s + (sc1 - t1s) - (mass_sum1 - s1s) * c2[j]);
    }
    prv = dist;
  }
  return ans;
}


vector<vector<ld>> compress(vector<vector<ld>> d){
  int n = d.size();
  vector<ld> ord(0);
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      ord.push_back(d[i][j]);
    }
  }
  sort(ord.begin(), ord.end());
  ord.erase(unique(ord.begin(), ord.end()), ord.end());
  vector<vector<ld>> r(n, vector<ld>(n));
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      r[i][j] = (int)(lower_bound(ord.begin(), ord.end(), d[i][j]) - ord.begin()) + 1;
      r[i][j] /= ord.size();
    }
  }
  return r;
}

ld robust_anchor_energy(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2){
  return anchor_energy(compress(d1), a1, compress(d2), a2);
}

pair<ld, vector<vector<ld>>> anchor_wasserstein(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2, ld eps=1){
  int n1 = a1.size();
  int n2 = a2.size();
  vector<vector<pair<ld, int>>> di1(n1);
  vector<vector<pair<ld, int>>> di2(n2);
  for(int i = 0; i < n1; i++){
    di1[i].clear();
    for(int k = 0; k < n1; k++){
      di1[i].emplace_back(d1[i][k], k);
    }
    sort(di1[i].begin(), di1[i].end());
  }
  for(int j = 0; j < n2; j++){
    di2[j].clear();
    for(int k = 0; k < n2; k++){
      di2[j].emplace_back(d2[j][k], k);
    }
    sort(di2[j].begin(), di2[j].end());
  }
  ld ans = 0;
  vector<vector<ld>> C(n1, vector<ld>(n2, 0));
  vector<vector<ld>> lK(n1, vector<ld>(n2, 0));
  for(int i = 0; i < n1; i++){
    for(int j = 0; j < n2; j++){
      ld c1 = 0;
      ld c2 = 0;
      int it1 = 0;
      int it2 = 0;
      ld prv = 0;
      while(it1 < n1 || it2 < n2){
        if(it1 == n1 || (it2 < n2 && di1[i][it1].first > di2[j][it2].first)){
          C[i][j] += a1[i] * a2[j] * (di2[j][it2].first - prv) * abs(c1 - c2);
          c2 += a2[di2[j][it2].second];
          prv = di2[j][it2].first;
          it2++;
        } else {
          C[i][j] += a1[i] * a2[j] * (di1[i][it1].first - prv) * abs(c1 - c2);
          c1 += a1[di1[i][it1].second];
          prv = di1[i][it1].first;
          it1++;
        }
      }
      lK[i][j] = -C[i][j] / eps;
    }
  }
  vector<vector<ld>> P = log_sinkhorn(lK, a1, a2);
  ld value = 0;
  for(int i = 0; i < n1; i++){
    for(int j = 0; j < n2; j++){
      value += P[i][j] * C[i][j] + eps * P[i][j] * (log(max(P[i][j], 1e-18)) - 1);
    }
  }
  return make_pair(value, P);
}

pair<ld, vector<vector<ld>>> robust_anchor_wasserstein(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2, ld eps=1){
  return anchor_wasserstein(compress(d1), a1, compress(d2), a2, eps);
}
