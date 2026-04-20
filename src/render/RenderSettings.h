#pragma once

struct RenderSettings
{
    int samplesPerPixel = 4;
    int maxBounces = 8;
    float exposure = 1.0f;
    bool debugDirectional = false;
    int debugMode = 0;
};
