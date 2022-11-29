#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

namespace netpbm {

    typedef unsigned char tupltype_t;

    const tupltype_t TUPLTYPE_GRAYSCALE = 1;
    const tupltype_t TUPLTYPE_RGB = 2;

    const std::string file_extensions[] = {".pgm", ".ppm"};
    const std::string magic_numbers[] = {"P5", "P6"};

    void NETPBM_ERROR(std::string name, std::string reason) {
        printf("\nNetpbm error: \"%s\": %s ... in \"%s\" on line %d\n", name.c_str(), reason.c_str(), __FILE__, __LINE__);
        exit(1);
    }

    bool verify_ext(std::string name, tupltype_t type) {
        std::string ext = name.substr(name.find_last_of("."));
        return ext == file_extensions[type - 1];
    }

    float* to_float(uint8_t* bytes, size_t size, uint8_t max) {
        float* out = (float*)malloc(size * sizeof(float));
        for (size_t i = 0; i < size; i++) out[i] = float(bytes[i]) / max;
        return out;
    }

    uint8_t* to_uint8(float* data, size_t size, uint8_t max) {
        uint8_t* out = (uint8_t*)malloc(size);
        for (size_t i = 0; i < size; i++) out[i] = uint8_t(data[i] * max);
        return out;
    }

    float* load_image(std::string name, int* width, int* height, tupltype_t* type) {
        std::ifstream ifs;
        ifs.open(name, std::ios::binary);
        if(!ifs.is_open()) {
            NETPBM_ERROR(name, "Couldn't open file");
        }

        std::string line;
        if (!std::getline(ifs, line)) {
            NETPBM_ERROR(name, "File empty");
        }

        if (line == "P5") {
            *type = TUPLTYPE_GRAYSCALE;
        } else if (line == "P6") {
            *type = TUPLTYPE_RGB;
        } else {
            NETPBM_ERROR(name, "Invalid header");
        }

        if (!verify_ext(name, *type)) {
            NETPBM_ERROR(name, "Tupltype and file extension mismatched");
        }

        size_t max;
        int numbers_found = 0;

        while (std::getline(ifs, line)) {
            if (line[0] == '#') {
                continue;
            }

            std::stringstream ss(line);
            std::string token;

            while (std::getline(ss, token, ' ')) {
                if (numbers_found == 0) {
                    *width = std::stoi(token);
                } else if (numbers_found == 1) {
                    *height = std::stoi(token);
                } else if (numbers_found == 2) {
                    max = std::stoi(token);
                } else {
                    NETPBM_ERROR(name, "Too many numbers found");
                }

                numbers_found++;
            }

            if (numbers_found == 3) {
                break;
            }
        }

        if (*width <= 0 || *height <= 0) {
            NETPBM_ERROR(name, "Invalid image dimensions");
        }

        if (max <= 0 || max > 255) {
            NETPBM_ERROR(name, "Invalid max value");
        }

        size_t size = *width * *height;
        if (*type == TUPLTYPE_RGB) {
            size *= 3;
        }

        uint8_t* bytes = (uint8_t*)malloc(size);
        ifs.read((char *)bytes, size);
        ifs.close();

        float* data = to_float(bytes, size, max);
        free(bytes);
        
        return data;
    }

    void save_image(std::string name, float* data, int width, int height, tupltype_t type) {
        if (!verify_ext(name, type)) {
            NETPBM_ERROR(name, "Tupltype and file extension mismatched");
        }

        size_t size = width * height;
        if (type == TUPLTYPE_RGB) {
            size *= 3;
        }
        uint8_t* bytes = to_uint8(data, size, 255);

        std::ofstream ofs;
        ofs.open(name, std::ios::binary);
        if(!ofs.is_open()) {
            NETPBM_ERROR(name, "Couldn't open file");
        }

        ofs << magic_numbers[type - 1] << std::endl
            << width << " " << height << std::endl
            << 255 << std::endl;
        ofs.write((char*)bytes, size);
        ofs.close();

        free(bytes);
    }
}