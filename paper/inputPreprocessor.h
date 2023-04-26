/*
 * inputPreprocessor.h
 * Prototypes for molecular preprocessing for PAPER 
 *  
 * Author: Imran Haque, 2009
 * Copyright 2009, Stanford University
 *
 * This file is licensed under the terms of the GPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */
#ifndef _INPUTPREPROCESSOR_H_
#define _INPUTPREPROCESSOR_H_

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

/* Defines an undirected graph with integer vertices */ 
class molgraph {
    public:
    void add_vertex(int v) {
        adj_list[v]=set<int>();
    }
    void add_edge(int u,int v) {
        adj_list[u].insert(v);
        adj_list[v].insert(u);
    }
    unsigned int num_vertices() const {
        return adj_list.size();
    }
    list<int> vertices() const {
        list<int> verts;
        for (map<int,set<int> >::const_iterator viter = adj_list.begin(); viter != adj_list.end(); viter++) {
            verts.push_back(viter->first);
        }
        return verts;
    }

    set<int> neighbors(int v) const {
        return (adj_list.find(v))->second;
    }
    string toDOT() const {
        stringstream buf;
        buf << "graph molgraph {" << endl;
        for (map<int,set<int> >::const_iterator viter = adj_list.begin(); viter != adj_list.end(); viter++) {
            buf << "n"<<(viter->first) << " [label=\""<<(viter->first)<<"\"];" << endl;
        }
        for (map<int,set<int> >::const_iterator viter = adj_list.begin(); viter != adj_list.end(); viter++) {
            for (set<int>::iterator eiter = viter->second.begin(); eiter != viter->second.end(); eiter++) {
                buf << "n"<<viter->first << " -- n" << *eiter << ";" << endl;
            }
        }
        buf << "}" << endl;
        return buf.str();
    }
    
    void shrink() {
        while (exists_degree_one()) {
            list<int> verticesToRemove;
            for (map<int,set<int> >::iterator i = adj_list.begin(); i!= adj_list.end(); i++) {
               if (i->second.size() == 1) verticesToRemove.push_back(i->first);
            }
            assert(verticesToRemove.size() > 0);
            for(list<int>::iterator i=verticesToRemove.begin(); i != verticesToRemove.end(); i++) {
                delete_vertex(*i);
            }
        }
    }
    protected:
    map<int,set<int> > adj_list;
    bool exists_degree_one() {
        for (map<int,set<int> >::iterator i = adj_list.begin(); i!= adj_list.end(); i++) {
            if (i->second.size() == 1)
                return true;
        }
        return false;
    }

    void delete_vertex(int v) {
        adj_list.erase(v);
        for (map<int,set<int> >::iterator i = adj_list.begin(); i!= adj_list.end(); i++) {
            i->second.erase(v);
        }
        return;
    }


};

list<set<int> > cyclic_decomposition(const molgraph& molecule);

list<set<int> > find_ring_systems_rdkit(RDKit::ROMol* rdmol);

#endif
