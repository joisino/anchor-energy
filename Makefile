distance_comparison: distance_comparison.cpp util.cpp anchor_wasserstein.cpp gromov.cpp util.hpp
	g++ -o $@ distance_comparison.cpp anchor_wasserstein.cpp gromov.cpp util.cpp -std=gnu++11 -O2

matching: matching.cpp util.cpp anchor_wasserstein.cpp gromov.cpp util.hpp
	g++ -o $@ matching.cpp anchor_wasserstein.cpp gromov.cpp util.cpp -std=gnu++11 -O2
