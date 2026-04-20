#pragma once

#include "Scene.h"

struct aiTexture;

class TextureRepository
{
public:
    static TextureData loadTexture(const std::string& path);
    static TextureData loadEmbeddedTexture(const aiTexture* aiTex);
    static bool isEmbeddedTextureRef(const std::string& path);
};
