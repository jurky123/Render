#!/usr/bin/env python3
"""
Convert OpenEXR (.exr) images to Radiance HDR (.hdr) format.
Required by PathTracer's HDR texture loader (stb_image based).

Usage:
    python convert_exr_to_hdr.py input.exr output.hdr
    
    # Or convert all .exr files in a directory:
    for f in *.exr; do python convert_exr_to_hdr.py "$f" "${f%.exr}.hdr"; done

Installation:
    1. Using OpenEXR Python bindings (easiest):
       conda install OpenEXR (or: pip install OpenEXR)
       
    2. Using Pillow with FreeImage plugin:
       pip install Pillow-SIMD
       
    3. Manual ImageMagick installation:
       Windows: Download from https://imagemagick.org/
       macOS:   brew install imagemagick
       Linux:   apt-get install imagemagick
       Then use: magick input.exr output.hdr
"""

import sys
import struct
import io

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False

def save_hdr(filename, width, height, pixels):
    """Save pixels as Radiance HDR (.hdr) format.
    
    Args:
        filename: Output .hdr filename
        width, height: Image dimensions
        pixels: List of (R, G, B) floats, row-major order
    """
    with open(filename, 'wb') as f:
        # Radiance HDR header
        f.write(b'#?RADIANCE\n')
        f.write(b'# Converted from OpenEXR\n')
        f.write(b'FORMAT=32-bit_rle_rgbe\n')
        f.write(b'EXPOSURE= 1.0\n')
        f.write(f'-Y {height} +X {width}\n'.encode())
        
        # Encode pixels as RGBE (Radiance HDR format)
        for y in range(height):
            scanline = []
            for x in range(width):
                idx = (y * width + x) * 3
                if idx + 2 < len(pixels):
                    r, g, b = float(pixels[idx]), float(pixels[idx+1]), float(pixels[idx+2])
                else:
                    r, g, b = 0.0, 0.0, 0.0
                
                # Convert RGB float to RGBE (Radiance HDR format)
                v = max(r, max(g, b))
                if v < 1e-32:
                    scanline.extend([0, 0, 0, 0])
                else:
                    m = v
                    e = -127
                    while m < 0.5:
                        m *= 2.0
                        e -= 1
                    while m >= 1.0:
                        m *= 0.5
                        e += 1
                    
                    mant_exp = int(e) + 128
                    if mant_exp < 0 or mant_exp > 255:
                        mant_exp = 128  # Fallback for out-of-range exponents
                    
                    scanline.append(max(0, min(255, int(r / m * 255.0 + 0.5))))
                    scanline.append(max(0, min(255, int(g / m * 255.0 + 0.5))))
                    scanline.append(max(0, min(255, int(b / m * 255.0 + 0.5))))
                    scanline.append(mant_exp)
            
            # Write minimal RLE encoding (uncompressed for simplicity)
            f.write(bytes(scanline))


def convert_exr_to_hdr(exr_filename, hdr_filename):
    """Convert OpenEXR to HDR format."""
    
    if not HAS_OPENEXR:
        print(f"ERROR: OpenEXR Python package not found.")
        print(f"Install it with: conda install OpenEXR")
        print(f"or: pip install OpenEXR")
        return False
    
    try:
        # Open EXR file
        exr_file = OpenEXR.InputFile(exr_filename)
        dw = exr_file.header()['dataWindow']
        size = exr_file.header()['displayWindow']
        
        width = int(dw.max.x) - int(dw.min.x) + 1
        height = int(dw.max.y) - int(dw.min.y) + 1
        
        print(f"Reading {exr_filename}: {width}x{height}")
        
        # Read RGB channels
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        
        pixels_r = exr_file.channel('R', pt)
        pixels_g = exr_file.channel('G', pt)
        pixels_b = exr_file.channel('B', pt)
        
        # Interleave into RGB format
        pixels = []
        for i in range(width * height):
            pixels.append(pixels_r[i])
            pixels.append(pixels_g[i])
            pixels.append(pixels_b[i])
        
        # Save as HDR
        save_hdr(hdr_filename, width, height, pixels)
        print(f"✓ Saved {hdr_filename}")
        return True
        
    except Exception as e:
        print(f"ERROR converting {exr_filename}: {e}")
        return False


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        print("Usage: python convert_exr_to_hdr.py <input.exr> <output.hdr>")
        sys.exit(1)
    
    exr_file = sys.argv[1]
    hdr_file = sys.argv[2]
    
    if convert_exr_to_hdr(exr_file, hdr_file):
        sys.exit(0)
    else:
        sys.exit(1)
