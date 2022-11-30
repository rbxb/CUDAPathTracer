#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "object.h"

namespace obj {

    struct Object {
        std::vector<float> verts;
        std::vector<int> indices;

        VerticesObject toVerticesObject();
    };

    VerticesObject Object::toVerticesObject() {
        VerticesObject ret;
        ret.verts = (Point*)malloc(indices.size() * sizeof(Point));
        ret.n = indices.size();

        for (int i = 0; i < indices.size(); i++) {
            int index = indices[i] - 1;
            memcpy(&ret.verts[i], &verts[index * 3], sizeof(Point));
        }

        return ret;
    }

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

    std::vector<Object> read(std::string name) {
        Object object = {std::vector<float>(), std::vector<int>()};
        std::vector<Object> objects = {};

        std::ifstream ifs;
        ifs.open(name, std::ios::binary);
        if(!ifs.is_open()) {
            OBJ_ERROR(name, "Couldn't open file");
        }

        std::string line;
        while (std::getline(ifs, line)) {
            if (line[0] == '#') {
                continue;
            }

            std::stringstream ss(line);
            std::string token;
            std::string first;
            std::getline(ss, first, ' ');

            int fanCount = 0;
            int fanStart = object.indices.size();

            while (std::getline(ss, token, ' ')) {
                int slashPos = token.find('/');
                if (slashPos != std::string::npos) {
                    token = token.substr(0, slashPos);
                }

                token = trim(token);
                if (token.size() == 0) continue;

                if (first == "v") {
                    object.verts.push_back(std::stof(token));
                } else if (first == "f") {
                    if (fanCount > 2) {
                        object.indices.push_back(object.indices[fanStart]);
                        object.indices.push_back(object.indices[fanStart + 1]);
                    }
                    fanCount++;
                    object.indices.push_back(std::stoi(token));
                } else if (first == "o") {
                    if (object.indices.size() > 0) objects.push_back(object);
                    object = {std::vector<float>(), std::vector<int>()};
                }
            }
        }

        objects.push_back(object);
        return objects;
    }
}