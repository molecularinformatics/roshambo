#include <iostream>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <tuple>
#include <algorithm>

using namespace std;

const float CONSTANT_P = 2.828427124742344;
const float PI = 3.14159265358;

typedef unordered_map<int, float> IntFloatMap;
typedef unordered_map<int, unordered_map<int, float>> IntIntFloatMap;

double my_pow(double x, size_t n){
    double r = 1.0;
    while(n > 0){
        r *= x;
        --n;
    }
    return r;
}

float calc_single_overlap(vector<int> atom_inds, IntFloatMap alpha_dict, IntIntFloatMap cross_alpha_distance_dict) {
    int ss = atom_inds.size();
    float p = my_pow(CONSTANT_P, ss);
    float alpha = 0;
    for (int ind : atom_inds) {
        alpha += alpha_dict[ind];
    }
    float k = 0;
    for (int i = 0; i < ss; i++) {
        for (int j = i + 1; j < ss; j++) {
            k += cross_alpha_distance_dict[atom_inds[i]][atom_inds[j]];
        }
    }
    float k_exp = exp(-k /alpha);
    float res = p * k_exp * my_pow(PI / alpha, 1.5);
    return res;
}
