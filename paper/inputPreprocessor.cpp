/*
 * inputPreprocessor.cpp
 * Handles molecular preprocessing for PAPER 
 * (currently, cyclic decompositions)
 *  
 * Author: Imran Haque, 2009
 * Copyright 2009, Stanford University
 *
 * This file is licensed under the terms of the GPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */

#include "inputPreprocessor.h"
#include <list>
#include <utility>
#include <map>
#include <set>
#include <stack>
#include <cassert>
#include <string>
#include <sstream>
#include <GraphMol/ROMol.h>

using namespace std;
using namespace RDKit;

list<set<int> > cyclic_decomposition(const molgraph& molecule) {
    map<int,int> cycle_domain;
    map<int,bool> visited;
    list<int> vertices = molecule.vertices();
    for (list<int>::iterator iter = vertices.begin(); iter != vertices.end(); iter++) {
        cycle_domain[*iter] = -1;
        visited[*iter] = false;
    }
    int next_cycle_domain = 0;
    
    typedef pair<int,list<int> > DFStype;
    stack<DFStype> DFSstack;
    
    list<int> emptylist;
    DFStype start(vertices.front(),emptylist);
    DFSstack.push(start);

    while (!DFSstack.empty()) {
        DFStype cur = DFSstack.top();
        DFSstack.pop();
        int vertex = cur.first;
        list<int>& path = cur.second;
        //cout << "Visiting vertex "<<vertex<<" with predecessor path:";
        //for (list<int>::iterator i = path.begin(); i!=path.end();i++)
        //    cout << " "<<*i<<",";
        //cout << endl;

        if (visited[vertex]) {
            // If we've visited this vertex before, then we've found a cycle
            int cur_cycle_domain;
            list<int>::iterator iter = path.begin();
            // Scan the predecessor path for the first time we found this vertex
            while (iter != path.end() && *iter != vertex) iter++;
            if (iter == path.end()) {
                // If this vertex doesn't appear in our predecessor path, it's possible
                // we already labeled it in a previous pass
                // But if it's not in the path and it's unlabeled, then we have a problem...
                assert(cycle_domain[vertex] != -1);
                continue;
            }
            // OK, found the first time it showed up in this path
            if (cycle_domain[vertex] == -1) {
                // This cycle has not been previously labeled
                cur_cycle_domain = next_cycle_domain++;
            } else {
                // Merging into an existing cyclic component
                cur_cycle_domain = cycle_domain[vertex];
            }
            // Add everything in the path to the current cycle domain
            while (iter != path.end()) {
                cycle_domain[*iter] = cur_cycle_domain;
                //cout << "\tAdding vertex "<<*iter<<" to cycle domain "<<cur_cycle_domain<<endl;
                iter++;
            }
            //cout << endl;
        } else {
            // We haven't seen this vertex before
            visited[vertex] = true;
            int parent_vertex = path.back();
            set<int> neighbors = molecule.neighbors(vertex);
            for (set<int>::iterator iter = neighbors.begin(); iter != neighbors.end(); iter++) {
                int neighbor_vertex = *iter;
                if (neighbor_vertex == parent_vertex) continue; // Don't go back to the parent
                //cout << "\tAdding vertex "<<neighbor_vertex<<" onto the DFS stack"<<endl;
                // Clone the current path and add ourselves to the end
                list<int> neighbor_path(path);
                neighbor_path.push_back(vertex);
                DFSstack.push(DFStype(neighbor_vertex,neighbor_path));
            }
        }
    }
    
    list<set<int> > cycle_domain_elements;
    // Multiple passes over cycle_domain because I'm lazy
    for (int domain = 0; domain < next_cycle_domain; domain++) {
        set<int> domain_members;
        for (map<int,int>::iterator i = cycle_domain.begin(); i != cycle_domain.end(); i++)
            if (i->second == domain) domain_members.insert(i->first);
        cycle_domain_elements.push_back(domain_members);
    }
    return cycle_domain_elements;
}

 list<set<int> > find_ring_systems_rdkit(RDKit::ROMol* rdmol) {
    molgraph graph;
    for (unsigned int i = 0; i < rdmol->getNumAtoms(); ++i) {
        const auto& atom = rdmol->getAtomWithIdx(i);
        graph.add_vertex(atom->getIdx());
    }
    for (unsigned int i = 0; i < rdmol->getNumBonds(); ++i) {
        const auto& bond = rdmol->getBondWithIdx(i);
        if (bond->getBeginAtomIdx() > bond->getEndAtomIdx())
            continue;
        graph.add_edge(bond->getBeginAtomIdx(), bond->getEndAtomIdx());
    }
    return cyclic_decomposition(graph);
}

