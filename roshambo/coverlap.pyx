from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map


cdef extern from "overlap.cpp":
    float calc_single_overlap(vector[int], unordered_map[int, float],
                              unordered_map[int, unordered_map[int, float]])

def calc_single_overlap_cython(atom_inds, alpha_dict, cross_alpha_distance_dict):
    return calc_single_overlap(atom_inds, alpha_dict, cross_alpha_distance_dict)
