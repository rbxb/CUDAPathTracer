#include "point.h"

class SceneReader {
public:
    SceneReader(std::string name);

    std::vector<ExpandedObject> getObjects() {return objects;};
    std::vector<Material> getMaterials() {return materials;};
    std::vector<Texture> getTextures() {return textures;};

private:
    std::vector<ExpandedObject> objects;
    std::vector<Material> materials;
    std::vector<Texture> textures;
}

SceneReader::SceneReader(std::string name) {
    

}