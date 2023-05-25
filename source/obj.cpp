#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

#include "object.h"

namespace obj {   
    void OBJ_ERROR(std::string name, std::string reason) {
        printf("\nObj error: \"%s\": %s ... in \"%s\" on line %d\n", name.c_str(), reason.c_str(), __FILE__, __LINE__);
        exit(1);
    }

    // https://www.chriswirz.com/software/cpp-string-trim
    std::string trim(const std::string &str) {
        std::string s(str);
        while (s.size() > 0 && std::isspace(s.front())) s.erase(s.begin());
        while (s.size() > 0 && std::isspace(s.back())) s.pop_back();
        return s;
    }

    class ObjectReader {
    public:
        std::vector<IndexedObject> read(std::string name);
    
    private:
        void processLine(std::string line);
        int processToken(std::string token);
        void processFaceToken(std::string face_element);
        IndexedObject packObject();

        std::vector<IndexedObject> objects;

        std::string first;
        int token_line_count;

        int fanCount;
        int fanStart;

        std::vector<float> flt_verts;
        std::vector<float> flt_tex_coords;
        std::vector<float> flt_normals;

        std::vector<int> v_indices;
        std::vector<int> t_indices;
        std::vector<int> n_indices;

        const int FOUND_COMMENT = 1;
    };

    std::vector<IndexedObject> ObjectReader::read(std::string name) {
        std::vector<IndexedObject> objects = std::vector<IndexedObject>();

        std::vector<float> flt_verts = std::vector<float>();
        std::vector<float> flt_tex_coords = std::vector<float>();
        std::vector<float> flt_normals = std::vector<float>();

        std::vector<int> v_indices = std::vector<int>();
        std::vector<int> t_indices = std::vector<int>();
        std::vector<int> n_indices = std::vector<int>();

        std::ifstream ifs;
        ifs.open(name, std::ios::binary);
        if(!ifs.is_open()) {
            OBJ_ERROR(name, "Couldn't open file");
        }

        std::string line;
        while (std::getline(ifs, line)) {
            processLine(line);
        }

        objects.push_back(packObject());
        return objects;
    }

    void ObjectReader::processLine(std::string line) {
        std::string token;
        token_line_count = 0;
        fanCount = 0;
        fanStart = v_indices.size();

        line = trim(line);

        while (line.size() > 0) {
            size_t break_pos = line.find(' ');
            token = line.substr(0, break_pos);
            token = trim(token);
            if (token.size() > 0) {
                if (processToken(token) == FOUND_COMMENT) {
                    break;
                }
                token_line_count++;
            }
            
            if (break_pos != std::string::npos) line = line.substr(break_pos + 1);
            else break;
        }

        // add an extra zero to tex coords because I am using a 3D Point to store the 2D value
        if (first == "vt") flt_tex_coords.push_back(0); 
    }

    int ObjectReader::processToken(std::string token) {
        if (token[0] == '#') return FOUND_COMMENT;

        if (token_line_count == 0) {
            first = token;
        } else {
            if (first == "f") {
                processFaceToken(token);
            } else if (first == "v") {
                flt_verts.push_back(std::stof(token));
            } else if (first == "vt") {
                flt_tex_coords.push_back(std::stof(token));
            } else if (first == "vn") {
                flt_normals.push_back(std::stof(token));
            }
        }

        return 0;
    }

    void ObjectReader::processFaceToken(std::string face_element) {
        int face_element_count = 0;
        std::string subtoken;

        if (fanCount > 2) {
            v_indices.push_back(v_indices[fanStart]);
            v_indices.push_back(v_indices[fanStart + 1]);

            if (t_indices.size() > 0) {
                t_indices.push_back(t_indices[fanStart]);
                t_indices.push_back(t_indices[fanStart + 1]);
            }

            if (n_indices.size() > 0) {
                n_indices.push_back(n_indices[fanStart]);
                n_indices.push_back(n_indices[fanStart + 1]);
            }
        }
        fanCount++;

        while (face_element.size() > 0) {
            int break_pos = face_element.find('/');
            subtoken = face_element.substr(0, break_pos);
            subtoken = trim(subtoken);
            
            if (subtoken.size() > 0) {
                if (face_element_count == 0) v_indices.push_back(std::stoi(subtoken) - 1);
                else if (face_element_count == 1) t_indices.push_back(std::stoi(subtoken) - 1);
                else if (face_element_count == 2) n_indices.push_back(std::stoi(subtoken) - 1);
                face_element_count++;
            }
            
            if (break_pos != std::string::npos) face_element = face_element.substr(break_pos + 1);
            else break;
        }
    }

    IndexedObject ObjectReader::packObject() {
        IndexedObject object = IndexedObject();
        
        object.verts = (Point*)(&flt_verts[0]);
        object.tex_coords = (Point*)(&flt_tex_coords[0]);
        object.normals = (Point*)(&flt_normals[0]);
        object.v_indices = (int*)(&v_indices[0]);
        object.t_indices = (int*)(&t_indices[0]);
        object.n_indices = (int*)(&n_indices[0]);

        object.has_tex_coods = t_indices.size() > 0;
        object.has_normals = n_indices.size() > 0;

        object.n = v_indices.size();

        return object;
    }
}