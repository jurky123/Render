#include <iostream>
#include <string>
#include "Scene.h"
#include "PBRTLoader.h"

int main(int argc, char* argv[]) {
    Scene scene;
    
    std::string pbrtPath = "pbrt-v4-scenes/bistro/bibistro_boulangerie_large.pbrt";
    std::cout << "[Test] Loading PBRT scene: " << pbrtPath << "\n";
    
    if (PBRTLoader::load(pbrtPath, scene)) {
        std::cout << "[Test] Scene loaded successfully!\n";
        std::cout << "[Test] Scene has " << scene.materials().size() << " materials\n";
        
        // Print first few material colors
        const auto& materials = scene.materials();
        for (size_t i = 0; i < std::min(size_t(10), materials.size()); ++i) {
            const auto& mat = materials[i];
            std::cout << "  Material " << i << ": " << mat.name 
                      << " -> color (" << mat.baseColor.x << ", " 
                      << mat.baseColor.y << ", " << mat.baseColor.z << ")\n";
        }
    } else {
        std::cout << "[Test] Failed to load PBRT scene\n";
    }
    
    return 0;
}
