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
        ret.verts = (float*)malloc(indices.size() * sizeof(float) * 3);
        ret.n = indices.size();

        for (int i = 0; i < indices.size(); i++) {
            int index = indices[i];
            memcpy(&ret.verts[i * 3], &verts[index * 3], 3 * sizeof(float));
        }

        return ret;
    }

    void OBJ_ERROR(std::string name, std::string reason) {
        printf("\nObj error: \"%s\": %s ... in \"%s\" on line %d\n", name.c_str(), reason.c_str(), __FILE__, __LINE__);
        exit(1);
    }

    std::vector<Object> read(std::string name) {
        Object object = {std::vector<float>(), std::vector<int>()};
        std::vector<Object> objects = {};

        std::ifstream ifs;
        ifs.open("resources/cube.obj", std::ios::binary);
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

            while (std::getline(ss, token, ' ')) {
                int slashPos = token.find('/');
                if (slashPos != std::string::npos) {
                    token = token.substr(0, slashPos);
                }

                if (first == "v") {
                    object.verts.push_back(std::stof(token));
                } else if (first == "f") {
                    object.indices.push_back(std::stoi(token));
                } else if (first == "o") {
                    objects.push_back(object);
                    object = {std::vector<float>(), std::vector<int>()};
                }
            }
        }

        objects.push_back(object);
        return objects;
    }
}