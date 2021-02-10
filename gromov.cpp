#include <vector>
#include <algorithm>
#include <utility>
#include <cmath>
#include "util.hpp"

using namespace std;

vector<vector<ld>> matmul(vector<vector<ld>> a, vector<vector<ld>> b){
  int n1 = a.size();
  int n2 = b[0].size();
  int n3 = b.size();
  vector<vector<ld>> c(n1, vector<ld>(n2, 0));
  for(int i = 0; i < n1; i++){
    for(int k = 0; k < n3; k++){
      for(int j = 0; j < n2; j++){
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return c;
}

vector<vector<ld>> transpose(vector<vector<ld>> a){
  int n1 = a.size();
  int n2 = a[0].size();
  vector<vector<ld>> b(n2, vector<ld>(n1));
  for(int i = 0; i < n1; i++){
    for(int j = 0; j < n2; j++){
      b[j][i] = a[i][j];
    }
  }
  return b;
}

ld sum(vector<vector<ld>> a){
  int n1 = a.size();
  int n2 = a[0].size();
  ld ans = 0;
  for(int i = 0; i < n1; i++){
    for(int j = 0; j < n2; j++){
      ans += a[i][j];
    }
  }
  return ans;
}

vector<vector<ld>> sinkhorn(vector<vector<ld>> K, vector<ld> a1, vector<ld> a2){
  int n1 = a1.size();
  int n2 = a2.size();
  vector<ld> v1(n1, 1);
  vector<ld> v2(n2, 1);
  vector<vector<ld>> prv_gamma(n1, vector<ld>(n2, 0));
  while(1){
    {
      vector<ld> tmp(n2, 0);
      for(int i = 0; i < n2; i++){
        tmp[i] = v2[i] * a2[i];
      }
      for(int i = 0; i < n1; i++){
        v1[i] = 0;
      }
      for(int i = 0; i < n1; i++){
        for(int j = 0; j < n2; j++){
          v1[i] += K[i][j] * tmp[j];
        }
        v1[i] = 1 / v1[i];
      }
    }
    {
      vector<ld> tmp(n1, 0);
      for(int i = 0; i < n1; i++){
        tmp[i] = v1[i] * a1[i];
      }
      for(int i = 0; i < n2; i++){
        v2[i] = 0;
      }
      for(int i = 0; i < n2; i++){
        for(int j = 0; j < n1; j++){
          v2[i] += K[j][i] * tmp[j];
        }
        v2[i] = 1 / v2[i];
      }
    }
    vector<vector<ld>> gamma(n1, vector<ld>(n2, 0));
    for(int i = 0; i < n1; i++){
      for(int j = 0; j < n2; j++){
        gamma[i][j] = v1[i] * K[i][j] * v2[j];
      }
    }
    ld diff = 0;
    ld norm = 0;
    for(int i = 0; i < n1; i++){
      for(int j = 0; j < n2; j++){
        diff += pow(gamma[i][j] - prv_gamma[i][j], 2);
        norm += pow(gamma[i][j], 2);
      }
    }
    diff = sqrt(diff);
    norm = sqrt(norm);
    if(diff / norm < 1e-6 || std::isnan(norm)){
      return gamma;
    }
    prv_gamma = gamma;
  }
}


vector<vector<ld>> log_sinkhorn(vector<vector<ld>> lK, vector<ld> a1, vector<ld> a2){
  int n1 = a1.size();
  int n2 = a2.size();
  vector<ld> lv1(n1, 0);
  vector<ld> lv2(n2, 0);
  vector<vector<ld>> prv_gamma(n1, vector<ld>(n2, 0));
  while(1){
    {
      for(int i = 0; i < n1; i++){
        vector<ld> tmp(n2);
        ld ma;
        for(int j = 0; j < n2; j++){
          tmp[j] = lK[i][j] + log(a2[j]) + lv2[j];
          if(j == 0){
            ma = tmp[j];
          } else {
            ma = max(ma, tmp[j]);
          }
        }
        lv1[i] = 0;
        for(int j = 0; j < n2; j++){
          lv1[i] += exp(tmp[j] - ma);
        }
        lv1[i] = - (log(max(lv1[i], 1e-18)) + ma);
      }
    }
    {
      for(int i = 0; i < n2; i++){
        vector<ld> tmp(n1);
        ld ma;
        for(int j = 0; j < n1; j++){
          tmp[j] = lK[j][i] + log(a1[j]) + lv1[j];
          if(j == 0){
            ma = tmp[j];
          } else {
            ma = max(ma, tmp[j]);
          }
        }
        lv2[i] = 0;
        for(int j = 0; j < n1; j++){
          lv2[i] += exp(tmp[j] - ma);
        }
        lv2[i] = - (log(max(lv2[i], 1e-18)) + ma);
      }
    }
    vector<vector<ld>> gamma(n1, vector<ld>(n2, 0));
    for(int i = 0; i < n1; i++){
      for(int j = 0; j < n2; j++){
        gamma[i][j] = exp(lv1[i] + lK[i][j] + lv2[j]);
      }
    }
    ld diff = 0;
    ld norm = 0;
    for(int i = 0; i < n1; i++){
      for(int j = 0; j < n2; j++){
        diff += pow(gamma[i][j] - prv_gamma[i][j], 2);
        norm += pow(gamma[i][j], 2);
      }
    }
    diff = sqrt(diff);
    norm = sqrt(norm);
    if(diff / norm < 1e-6 || std::isnan(norm)){
      return gamma;
    }
    prv_gamma = gamma;
  }
}


pair<ld, vector<vector<ld>>> gromov_sinkhorn(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2, ld alpha=1, ld eta=0.1){
  int n1 = a1.size();
  int n2 = a2.size();
  vector<vector<ld>> nd1 = d1;
  for(int i = 0; i < n1; i++){
    for(int j = 0; j < n1; j++){
      nd1[i][j] *= a1[j];
    }
  }
  vector<vector<ld>> nd2 = d2;
  for(int i = 0; i < n2; i++){
    for(int j = 0; j < n2; j++){
      nd2[i][j] *= a2[j];
    }
  }
  vector<vector<ld>> gamma(n1, vector<ld>(n2, 1));
  ld ans_prv = 1e18;
  vector<vector<ld>> K = matmul(nd1, matmul(gamma, transpose(nd2)));
  while(1){
    vector<vector<ld>> ngamma = gamma;
    for(int i = 0; i < n1; i++){
      for(int j = 0; j < n2; j++){
        ngamma[i][j] = K[i][j] * eta / alpha + log(max(gamma[i][j], 1e-18)) * (1 - eta);
      }
    }
    auto prv_gamma = gamma;
    gamma = log_sinkhorn(ngamma, a1, a2);
    {
      ld ans = 0;
      for(int i = 0; i < n1; i++){
        for(int j = 0; j < n1; j++){
          ans += d1[i][j] * d1[i][j] * a1[i] * a1[j];
        }
      }
      for(int i = 0; i < n2; i++){
        for(int j = 0; j < n2; j++){
          ans += d2[i][j] * d2[i][j] * a2[i] * a2[j];
        }
      }
      K = matmul(nd1, matmul(gamma, transpose(nd2)));
      for(int i = 0; i < n1; i++){
        for(int j = 0; j < n2; j++){
          ans -= 2 * K[i][j] * gamma[i][j] * a1[i] * a2[j];
        }
      }
      for(int i = 0; i < n1; i++){
        for(int j = 0; j < n2; j++){
          ans += alpha * gamma[i][j] * log(max(gamma[i][j], 1e-18)) * a1[i] * a2[j];
        }
      }
      if((ans_prv - ans) / ans < 1e-6){
        break;
      }
      ans_prv = ans;
    }
    {
      ld diff = 0;
      ld norm = 0;
      for(int i = 0; i < n1; i++){
        for(int j = 0; j < n2; j++){
          diff += pow(gamma[i][j] - prv_gamma[i][j], 2);
          norm += pow(gamma[i][j], 2);
        }
      }
      diff = sqrt(diff);
      norm = sqrt(norm);
      if(diff / norm < 1e-6 || std::isnan(norm)){
        break;
      }
    }
  }
  return make_pair(ans_prv, gamma);
}
