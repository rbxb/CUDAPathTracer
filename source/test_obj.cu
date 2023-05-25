#include "constants.h"
#include "obj.cpp"
#include "object.h"

#include <stdio.h>

const std::string testFile = "resources/cube-tex.obj";

int main(int argc, char* argv[]) {
    obj::ObjectReader reader = obj::ObjectReader();
    std::vector<IndexedObject> objects = reader.read(testFile);
    return 0;
}