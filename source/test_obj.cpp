#include "obj.cpp"

#include <stdio.h>

const std::string testFile = "resources/cube.obj";

bool testObj() {
    std::vector<obj::Object> objects = obj::read(testFile);
    return true;
}