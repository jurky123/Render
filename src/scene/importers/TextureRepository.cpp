#include "TextureRepository.h"

#include <assimp/scene.h>

#include <cstdlib>
#include <iostream>

TextureData TextureRepository::loadTexture(const std::string& path)
{
    TextureData tex;
    if (path.empty())
        return tex;

    stbi_set_flip_vertically_on_load(1);
    int origChannels = 0;
    const int desiredChannels = 4;
    tex.pixels = stbi_load(path.c_str(), &tex.width, &tex.height, &origChannels, desiredChannels);
    tex.channels = tex.pixels ? desiredChannels : 0;
    tex.ownsMemory = true;

    if (!tex.pixels)
        std::cerr << "[Scene] Failed to load texture: " << path << "\n";
    return tex;
}

bool TextureRepository::isEmbeddedTextureRef(const std::string& path)
{
    return !path.empty() && path[0] == '*';
}

TextureData TextureRepository::loadEmbeddedTexture(const aiTexture* aiTex)
{
    TextureData tex;
    if (!aiTex)
        return tex;

    const int desiredChannels = 4;
    if (aiTex->mHeight == 0)
    {
        const unsigned char* data = reinterpret_cast<const unsigned char*>(aiTex->pcData);
        const int dataSize = static_cast<int>(aiTex->mWidth);
        stbi_set_flip_vertically_on_load(1);
        int origChannels = 0;
        tex.pixels = stbi_load_from_memory(data, dataSize, &tex.width, &tex.height, &origChannels, desiredChannels);
        tex.channels = tex.pixels ? desiredChannels : 0;
        tex.ownsMemory = true;
    }
    else
    {
        tex.width = static_cast<int>(aiTex->mWidth);
        tex.height = static_cast<int>(aiTex->mHeight);
        tex.channels = desiredChannels;
        tex.ownsMemory = true;
        const size_t byteCount = static_cast<size_t>(tex.width) * tex.height * desiredChannels;
        tex.pixels = static_cast<unsigned char*>(std::malloc(byteCount));
        if (!tex.pixels)
        {
            tex.channels = 0;
            return tex;
        }

        for (size_t i = 0; i < static_cast<size_t>(tex.width) * tex.height; ++i)
        {
            const aiTexel& t = aiTex->pcData[i];
            tex.pixels[i * 4 + 0] = t.r;
            tex.pixels[i * 4 + 1] = t.g;
            tex.pixels[i * 4 + 2] = t.b;
            tex.pixels[i * 4 + 3] = t.a;
        }
    }

    if (!tex.pixels)
        std::cerr << "[Scene] Failed to load embedded texture (compressed=" << (aiTex->mHeight == 0) << ")\n";
    return tex;
}
