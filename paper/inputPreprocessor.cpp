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
#include <openbabel/obconversion.h>
#include <openbabel/mol.h>
#include <openbabel/obiter.h>
#include <openbabel/data.h>


using namespace std;
using namespace OpenBabel;

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

 list<set<int> > find_ring_systems(OBMol& mol) {
    molgraph graph;
    for (OBMolAtomIter a(mol); a; a++) {
        //cout << "Found atom: "<<a->GetIdx()<<" with atomic num = "<<a->GetAtomicNum()<<" and VdW radius "<<etab.GetVdwRad(a->GetAtomicNum())<<endl;
        graph.add_vertex(a->GetIdx());
    }
    for (OBMolBondIter b(mol); b; b++) {
        //cout << "Found bond:"<<b->GetBeginAtom()->GetIdx()<<"-"<<b->GetEndAtom()->GetIdx()<<endl;
        if (b->GetBeginAtom()->GetIdx() > b->GetEndAtom()->GetIdx())
            continue;
        //cout << "Adding edge:"<<b->GetBeginAtom()->GetIdx()<<"-"<<b->GetEndAtom()->GetIdx()<<endl;
        graph.add_edge(b->GetBeginAtom()->GetIdx(),b->GetEndAtom()->GetIdx());
    }
    //graph.shrink();
    //cout << graph.toDOT();
    return cyclic_decomposition(graph);
}

static int test_main(int argc,char** argv) {
    OBConversion conv;
    OBFormat* inFormat = conv.FormatFromExt(argv[1]);
    cout << "setinformat: "<< conv.SetInFormat(inFormat)<<endl;
    OBMol mol;
    cout << "readfile "<<conv.ReadFile(&mol,argv[1])<<endl;
    list<set<int> > ringsystems = find_ring_systems(mol);
    cout << "Found ring systems: "<<endl;
    for (list<set<int> >::iterator iter = ringsystems.begin(); iter!=ringsystems.end(); iter++) {
        for (set<int>::iterator it2 = iter->begin();  it2 != iter->end(); it2++) {
            cout << *it2 << " ";
        }
        cout << endl;
    }
    return 0;
}
