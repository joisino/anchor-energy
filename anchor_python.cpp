#include <vector>
#include <iostream>
#include <boost/python/numpy.hpp>
#include "util.hpp"

using namespace std;

namespace py = boost::python;
namespace np = boost::python::numpy;

using ld = double;

ld anchor_energy(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2);
ld robust_anchor_energy(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2);
pair<ld, vector<vector<ld>>> anchor_energy_naive(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2);
pair<ld, vector<vector<ld>>> anchor_wasserstein(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2, ld eps=1);
pair<ld, vector<vector<ld>>> robust_anchor_wasserstein(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2, ld eps=1);
pair<ld, vector<vector<ld>>> gromov_sinkhorn(vector<vector<ld>> d1, vector<ld> a1, vector<vector<ld>> d2, vector<ld> a2, ld alpha=1, ld eta=0.1);

tuple<vector<vector<ld>>, vector<ld>, vector<vector<ld>>, vector<ld>> convert(np::ndarray d1, np::ndarray a1, np::ndarray d2, np::ndarray a2){
  assert(d1.get_nd() == 2);
  assert(d2.get_nd() == 2);
  int n = d1.shape(0);
  assert(a1.shape(0) == n);
  assert(d1.shape(1) == n);
  int m = d2.shape(0);
  assert(a2.shape(0) == m);
  assert(d2.shape(1) == m);
  vector<vector<ld>> d1v(n, vector<ld>(n)), d2v(m, vector<ld>(m));
  vector<ld> a1v(n), a2v(m);
  for(int i = 0; i < n; i++){
    a1v[i] = py::extract<double>(a1[i]);
    for(int j = 0; j < n; j++){
      d1v[i][j] = py::extract<double>(d1[i][j]);
    }
  }
  for(int i = 0; i < m; i++){
    a2v[i] = py::extract<double>(a2[i]);
    for(int j = 0; j < m; j++){
      d2v[i][j] = py::extract<double>(d2[i][j]);
    }
  }
  return make_tuple(d1v, a1v, d2v, a2v);
}

np::ndarray to_ndarray(vector<vector<ld>> a){
  py::tuple shapeA = py::make_tuple(a.size(), a[0].size());
  np::ndarray A = np::zeros(shapeA, np::dtype::get_builtin<double>());
  for(int i = 0; i < a.size(); i++){
    for(int j = 0; j < a[i].size(); j++){
      A[i][j] = a[i][j];
    }
  }
  return A;
}

double anchor_energy_wrapper(np::ndarray d1, np::ndarray a1, np::ndarray d2, np::ndarray a2){
  vector<vector<ld>> d1v, d2v;
  vector<ld> a1v, a2v;
  tie(d1v, a1v, d2v, a2v) = convert(d1, a1, d2, a2);
  return anchor_energy(d1v, a1v, d2v, a2v);
}

double robust_anchor_energy_wrapper(np::ndarray d1, np::ndarray a1, np::ndarray d2, np::ndarray a2){
  vector<vector<ld>> d1v, d2v;
  vector<ld> a1v, a2v;
  tie(d1v, a1v, d2v, a2v) = convert(d1, a1, d2, a2);
  return robust_anchor_energy(d1v, a1v, d2v, a2v);
}

double anchor_wasserstein_wrapper(np::ndarray d1, np::ndarray a1, np::ndarray d2, np::ndarray a2, double eps){
  vector<vector<ld>> d1v, d2v;
  vector<ld> a1v, a2v;
  tie(d1v, a1v, d2v, a2v) = convert(d1, a1, d2, a2);
  return anchor_wasserstein(d1v, a1v, d2v, a2v, eps).first;
}

double robust_anchor_wasserstein_wrapper(np::ndarray d1, np::ndarray a1, np::ndarray d2, np::ndarray a2, double eps){
  vector<vector<ld>> d1v, d2v;
  vector<ld> a1v, a2v;
  tie(d1v, a1v, d2v, a2v) = convert(d1, a1, d2, a2);
  return robust_anchor_wasserstein(d1v, a1v, d2v, a2v, eps).first;
}

double gromov_wasserstein_wrapper(np::ndarray d1, np::ndarray a1, np::ndarray d2, np::ndarray a2, double alpha, double eta){
  vector<vector<ld>> d1v, d2v;
  vector<ld> a1v, a2v;
  tie(d1v, a1v, d2v, a2v) = convert(d1, a1, d2, a2);
  return gromov_sinkhorn(d1v, a1v, d2v, a2v, alpha, eta).first;
}



np::ndarray anchor_energy_matching_wrapper(np::ndarray d1, np::ndarray a1, np::ndarray d2, np::ndarray a2){
  vector<vector<ld>> d1v, d2v;
  vector<ld> a1v, a2v;
  tie(d1v, a1v, d2v, a2v) = convert(d1, a1, d2, a2);
  return to_ndarray(anchor_energy_naive(d1v, a1v, d2v, a2v).second);
}

np::ndarray anchor_wasserstein_matching_wrapper(np::ndarray d1, np::ndarray a1, np::ndarray d2, np::ndarray a2, double eps){
  vector<vector<ld>> d1v, d2v;
  vector<ld> a1v, a2v;
  tie(d1v, a1v, d2v, a2v) = convert(d1, a1, d2, a2);
  return to_ndarray(anchor_wasserstein(d1v, a1v, d2v, a2v, eps).second);
}

np::ndarray gromov_wasserstein_matching_wrapper(np::ndarray d1, np::ndarray a1, np::ndarray d2, np::ndarray a2, double alpha, double eta){
  vector<vector<ld>> d1v, d2v;
  vector<ld> a1v, a2v;
  tie(d1v, a1v, d2v, a2v) = convert(d1, a1, d2, a2);
  return to_ndarray(gromov_sinkhorn(d1v, a1v, d2v, a2v, alpha, eta).second);
}



BOOST_PYTHON_MODULE(anchor){
  Py_Initialize();
  np::initialize();  
  boost::python::def("anchor_energy",
                     anchor_energy_wrapper,
                     boost::python::args("d1",
                                         "a1",
                                         "d2",
                                         "a2"),
                     "Fast computation of the Anchor Energy between MMS A and B\n"
                     "\n"
                     "Parameters\n"
                     "----------\n"
                     "d1 : ndarray of size (n, n)\n"
                     "    Lists of distances\n"
                     "    d1[i][j] is the distance between i and j in MMS A\n"
                     "a1 : ndarray of size (n,)\n"
                     "    A list of weights\n"
                     "    a1[i] is the weight of (mass on) i in MMS A\n"
                     "d2 : ndarray of size (m, m)\n"
                     "    Lists of distances\n"
                     "    d2[i][j] is the distance between i and j in MMS B\n"
                     "a2 : ndarray of size (m,)\n"
                     "    A list of weights\n"
                     "    a2[i] is the weight of (mass on) i in MMS B\n"
                     "\n"
                     "Returns\n"
                     "-------\n"
                     "AE: float\n"
                     "    The Anchor Energy\n");

  boost::python::def("robust_anchor_energy",
                     robust_anchor_energy_wrapper,
                     boost::python::args("d1",
                                         "a1",
                                         "d2",
                                         "a2"),
                     "Fast computation of the Robust Anchor Energy between MMS A and B\n"
                     "\n"
                     "Parameters\n"
                     "----------\n"
                     "d1 : ndarray of size (n, n)\n"
                     "    Lists of distances\n"
                     "    d1[i][j] is the distance between i and j in MMS A\n"
                     "a1 : ndarray of size (n,)\n"
                     "    A list of weights\n"
                     "    a1[i] is the weight of (mass on) i in MMS A\n"
                     "d2 : ndarray of size (m, m)\n"
                     "    Lists of distances\n"
                     "    d2[i][j] is the distance between i and j in MMS B\n"
                     "a2 : ndarray of size (m,)\n"
                     "    A list of weights\n"
                     "    a2[i] is the weight of (mass on) i in MMS B\n"
                     "\n"
                     "Returns\n"
                     "-------\n"
                     "RAE: float\n"
                     "    The Robust Anchor Energy\n");

  boost::python::def("anchor_wasserstein",
                     anchor_wasserstein_wrapper,
                     boost::python::args("d1",
                                         "a1",
                                         "d2",
                                         "a2",
                                         "eps"),
                     "The Anchor Wasserstein between MMS A and B\n"
                     "\n"
                     "Parameters\n"
                     "----------\n"
                     "d1 : ndarray of size (n, n)\n"
                     "    Lists of distances\n"
                     "    d1[i][j] is the distance between i and j in MMS A\n"
                     "a1 : ndarray of size (n,)\n"
                     "    A list of weights\n"
                     "    a1[i] is the weight of (mass on) i in MMS A\n"
                     "d2 : ndarray of size (m, m)\n"
                     "    Lists of distances\n"
                     "    d2[i][j] is the distance between i and j in MMS B\n"
                     "a2 : ndarray of size (m,)\n"
                     "    A list of weights\n"
                     "    a2[i] is the weight of (mass on) i in MMS B\n"
                     "eps : float\n"
                     "    The regularization coefficient of the Sinkhorn algorithm\n"
                     "\n"
                     "Returns\n"
                     "-------\n"
                     "AW: float\n"
                     "    The Anchor Wasserstein\n");

  boost::python::def("robust_anchor_wasserstein",
                     robust_anchor_wasserstein_wrapper,
                     boost::python::args("d1",
                                         "a1",
                                         "d2",
                                         "a2",
                                         "eps"),
                     "The Robust Anchor Wasserstein between MMS A and B\n"
                     "\n"
                     "Parameters\n"
                     "----------\n"
                     "d1 : ndarray of size (n, n)\n"
                     "    Lists of distances\n"
                     "    d1[i][j] is the distance between i and j in MMS A\n"
                     "a1 : ndarray of size (n,)\n"
                     "    A list of weights\n"
                     "    a1[i] is the weight of (mass on) i in MMS A\n"
                     "d2 : ndarray of size (m, m)\n"
                     "    Lists of distances\n"
                     "    d2[i][j] is the distance between i and j in MMS B\n"
                     "a2 : ndarray of size (m,)\n"
                     "    A list of weights\n"
                     "    a2[i] is the weight of (mass on) i in MMS B\n"
                     "eps : float\n"
                     "    The regularization coefficient of the Sinkhorn algorithm\n"
                     "\n"
                     "Returns\n"
                     "-------\n"
                     "RAW: float\n"
                     "    The Robust Anchor Wasserstein\n");

  boost::python::def("gromov_wasserstein",
                     gromov_wasserstein_wrapper,
                     boost::python::args("d1",
                                         "a1",
                                         "d2",
                                         "a2",
                                         "alpha",
                                         "eta"),
                     "The Robust Anchor Wasserstein between MMS A and B\n"
                     "\n"
                     "Parameters\n"
                     "----------\n"
                     "d1 : ndarray of size (n, n)\n"
                     "    Lists of distances\n"
                     "    d1[i][j] is the distance between i and j in MMS A\n"
                     "a1 : ndarray of size (n,)\n"
                     "    A list of weights\n"
                     "    a1[i] is the weight of (mass on) i in MMS A\n"
                     "d2 : ndarray of size (m, m)\n"
                     "    Lists of distances\n"
                     "    d2[i][j] is the distance between i and j in MMS B\n"
                     "a2 : ndarray of size (m,)\n"
                     "    A list of weights\n"
                     "    a2[i] is the weight of (mass on) i in MMS B\n"
                     "alpha : float\n"
                     "    The regularization coefficient\n"
                     "eta : float\n"
                     "    The momentum of optimization\n"
                     "\n"
                     "Returns\n"
                     "-------\n"
                     "GW: float\n"
                     "    The Gromov Wasserstein\n");


  boost::python::def("anchor_energy_matching",
                     anchor_energy_matching_wrapper,
                     boost::python::args("d1",
                                         "a1",
                                         "d2",
                                         "a2"),
                     "The Anchor Energy Matching between MMS A and B\n"
                     "\n"
                     "Parameters\n"
                     "----------\n"
                     "d1 : ndarray of size (n, n)\n"
                     "    Lists of distances\n"
                     "    d1[i][j] is the distance between i and j in MMS A\n"
                     "a1 : ndarray of size (n,)\n"
                     "    A list of weights\n"
                     "    a1[i] is the weight of (mass on) i in MMS A\n"
                     "d2 : ndarray of size (m, m)\n"
                     "    Lists of distances\n"
                     "    d2[i][j] is the distance between i and j in MMS B\n"
                     "a2 : ndarray of size (m,)\n"
                     "    A list of weights\n"
                     "    a2[i] is the weight of (mass on) i in MMS B\n"
                     "\n"
                     "Returns\n"
                     "-------\n"
                     "AEM: ndarray of size (n, m)\n"
                     "    The Anchor Energy Matching\n");

  boost::python::def("anchor_wasserstein_matching",
                     anchor_wasserstein_matching_wrapper,
                     boost::python::args("d1",
                                         "a1",
                                         "d2",
                                         "a2",
                                         "eps"),
                     "The Anchor Wasserstein Matching between MMS A and B\n"
                     "\n"
                     "Parameters\n"
                     "----------\n"
                     "d1 : ndarray of size (n, n)\n"
                     "    Lists of distances\n"
                     "    d1[i][j] is the distance between i and j in MMS A\n"
                     "a1 : ndarray of size (n,)\n"
                     "    A list of weights\n"
                     "    a1[i] is the weight of (mass on) i in MMS A\n"
                     "d2 : ndarray of size (m, m)\n"
                     "    Lists of distances\n"
                     "    d2[i][j] is the distance between i and j in MMS B\n"
                     "a2 : ndarray of size (m,)\n"
                     "    A list of weights\n"
                     "    a2[i] is the weight of (mass on) i in MMS B\n"
                     "eps : float\n"
                     "    The regularization coefficient of the Sinkhorn algorithm\n"
                     "\n"
                     "Returns\n"
                     "-------\n"
                     "AWM: ndarray of size (n, m)\n"
                     "    The Anchor Wasserstein Matching\n");

  boost::python::def("gromov_wasserstein_matching",
                     gromov_wasserstein_matching_wrapper,
                     boost::python::args("d1",
                                         "a1",
                                         "d2",
                                         "a2",
                                         "alpha",
                                         "eta"),
                     "The Gromov Wasserstein Matching between MMS A and B\n"
                     "\n"
                     "Parameters\n"
                     "----------\n"
                     "d1 : ndarray of size (n, n)\n"
                     "    Lists of distances\n"
                     "    d1[i][j] is the distance between i and j in MMS A\n"
                     "a1 : ndarray of size (n,)\n"
                     "    A list of weights\n"
                     "    a1[i] is the weight of (mass on) i in MMS A\n"
                     "d2 : ndarray of size (m, m)\n"
                     "    Lists of distances\n"
                     "    d2[i][j] is the distance between i and j in MMS B\n"
                     "a2 : ndarray of size (m,)\n"
                     "    A list of weights\n"
                     "    a2[i] is the weight of (mass on) i in MMS B\n"
                     "alpha : float\n"
                     "    The regularization coefficient\n"
                     "eta : float\n"
                     "    The momentum of optimization\n"
                     "\n"
                     "Returns\n"
                     "-------\n"
                     "GWM: ndarray of size (n, m)\n"
                     "    The Gromov Wasserstein Matching\n");
}
