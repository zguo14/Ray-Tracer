#pragma once
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX


// OpenGL
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>


// image loader and writer
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


// linear algebra 
#include "linalg.h"
using namespace linalg::aliases;


// animated GIF writer
#include "gif.h"


// misc
#include <iostream>
#include <vector>
#include <cfloat>
#include <chrono>
#include <random>
#include <cstdlib>  // for srand
#include <ctime>  // for time


// main window
static GLFWwindow* globalGLFWindow;


// window size and resolution
// // (do not make it too large - will be slow!)
// constexpr int globalWidth = 120;
// constexpr int globalHeight = 60;

// constexpr int globalWidth = 256;
// constexpr int globalHeight = 192;

// constexpr int globalWidth = 1024;
// constexpr int globalHeight = 768;

constexpr int globalWidth = 1324;
constexpr int globalHeight = 768;

// constexpr int globalWidth = 512;
// constexpr int globalHeight = 384;
 
// degree and radian
constexpr float PI = 3.14159265358979f;
constexpr float DegToRad = PI / 180.0f;
constexpr float RadToDeg = 180.0f / PI;


// for ray tracing
constexpr float Epsilon = 5e-5f;


// amount the camera moves with a mouse and a keyboard
constexpr float ANGFACT = 0.2f;
constexpr float SCLFACT = 0.1f;


// fixed camera parameters
constexpr float globalAspectRatio = float(globalWidth / float(globalHeight));
constexpr float globalFOV = 45.0f; // vertical field of view
constexpr float globalDepthMin = Epsilon; // for rasterization
constexpr float globalDepthMax = 100.0f; // for rasterization
constexpr float globalFilmSize = 0.032f; //for ray tracing
const float globalDistanceToFilm = globalFilmSize / (2.0f * tan(globalFOV * DegToRad * 0.5f)); // for ray tracing


// particle system related
bool globalEnableParticles = false;
constexpr float deltaT = 0.002f;
constexpr float3 globalGravity = float3(0.0f, -9.8f, 0.0f);
constexpr int globalNumParticles = 200;


// dynamic camera parameters
// default, for glossy reflection
// float3 globalEye = float3(0.0f, 0.0f, 1.5f);
// float3 globalLookat = float3(0.0f, 0.0f, 0.0f);

// for room corner
float3 globalEye = float3(1.514173f, 1.140000f, 2.581319f); // x: left right, y: up down, z: near far
float3 globalLookat = float3(0.934173f, 0.739000f, 1.611319f);

float3 globalUp = normalize(float3(0.0f, 1.0f, 0.0f));  
float3 globalViewDir; // should always be normalize(globalLookat - globalEye)
float3 globalRight; // should always be normalize(cross(globalViewDir, globalUp));
bool globalShowRaytraceProgress = false; // for ray tracing


// mouse event
static bool mouseLeftPressed;
static double m_mouseX = 0.0;
static double m_mouseY = 0.0;


// rendering algorithm
enum enumRenderType {
	RENDER_RASTERIZE,
	RENDER_RAYTRACE,
	RENDER_IMAGE,
};
enumRenderType globalRenderType = RENDER_IMAGE;
int globalFrameCount = 0;
static bool globalRecording = false;
static GifWriter globalGIFfile;
constexpr int globalGIFdelay = 1;


// OpenGL related data (do not modify it if it is working)
static GLuint GLFrameBufferTexture;
static GLuint FSDraw;
static const std::string FSDrawSource = R"(
    #version 120

    uniform sampler2D input_tex;
    uniform vec4 BufInfo;

    void main()
    {
        gl_FragColor = texture2D(input_tex, gl_FragCoord.st * BufInfo.zw);
    }
)";
static const char* PFSDrawSource = FSDrawSource.c_str();



// fast random number generator based pcg32_fast
#include <stdint.h>
namespace PCG32 {
	static uint64_t mcg_state = 0xcafef00dd15ea5e5u;	// must be odd
	static uint64_t const multiplier = 6364136223846793005u;
	uint32_t pcg32_fast(void) {
		uint64_t x = mcg_state;
		const unsigned count = (unsigned)(x >> 61);
		mcg_state = x * multiplier;
		x ^= x >> 22;
		return (uint32_t)(x >> (22 + count));
	}
	float rand() {
		return float(double(pcg32_fast()) / 4294967296.0);
	}
}



// image with a depth buffer
// (depth buffer is not always needed, but hey, we have a few GB of memory, so it won't be an issue...)
class Image {
public:
	std::vector<float3> pixels;
	std::vector<float> depths;
	int width = 0, height = 0;

	static float toneMapping(const float r) {
		// you may want to implement better tone mapping
		return std::max(std::min(1.0f, r), 0.0f);
	}

	static float gammaCorrection(const float r, const float gamma = 1.0f) {
		// assumes r is within 0 to 1
		// gamma is typically 2.2, but the default is 1.0 to make it linear
		return pow(r, 1.0f / gamma);
	}

	void resize(const int newWdith, const int newHeight) {
		this->pixels.resize(newWdith * newHeight);
		this->depths.resize(newWdith * newHeight);
		this->width = newWdith;
		this->height = newHeight;
	}

	void clear() {
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				this->pixel(i, j) = float3(0.0f);
				this->depth(i, j) = FLT_MAX;
			}
		}
	}

	Image(int _width = 0, int _height = 0) {
		this->resize(_width, _height);
		this->clear();
	}

	bool valid(const int i, const int j) const {
		return (i >= 0) && (i < this->width) && (j >= 0) && (j < this->height);
	}

	float& depth(const int i, const int j) {
		return this->depths[i + j * width];
	}

	float3& pixel(const int i, const int j) {
		// optionally can check with "valid", but it will be slow
		return this->pixels[i + j * width];
	}

	void load(const char* fileName) {
		int comp, w, h;
		float* buf = stbi_loadf(fileName, &w, &h, &comp, 3);
		if (!buf) {
			std::cerr << "Unable to load: " << fileName << std::endl;
			return;
		}

		this->resize(w, h);
		int k = 0;
		for (int j = height - 1; j >= 0; j--) {
			for (int i = 0; i < width; i++) {
				this->pixels[i + j * width] = float3(buf[k], buf[k + 1], buf[k + 2]);
				k += 3;
			}
		}
		delete[] buf;
		printf("Loaded \"%s\".\n", fileName);
	}
	void save(const char* fileName) {
		unsigned char* buf = new unsigned char[width * height * 3];
		int k = 0;
		for (int j = height - 1; j >= 0; j--) {
			for (int i = 0; i < width; i++) {
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).x)));
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).y)));
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).z)));
			}
		}
		stbi_write_png(fileName, width, height, 3, buf, width * 3);
		delete[] buf;
		printf("Saved \"%s\".\n", fileName);
	}
};

// main image buffer to be displayed
Image FrameBuffer(globalWidth, globalHeight);

// you may want to use the following later for progressive ray tracing
Image AccumulationBuffer(globalWidth, globalHeight);
unsigned int sampleCount = 0;



// keyboard events (you do not need to modify it unless you want to)
void keyFunc(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS || action == GLFW_REPEAT) {
		switch (key) {
			case GLFW_KEY_R: {
				if (globalRenderType == RENDER_RAYTRACE) {
					printf("(Switched to rasterization)\n");
					glfwSetWindowTitle(window, "Rasterization mode");
					globalRenderType = RENDER_RASTERIZE;
				} else if (globalRenderType == RENDER_RASTERIZE) {
					printf("(Switched to ray tracing)\n");
					AccumulationBuffer.clear();
					sampleCount = 0;
					glfwSetWindowTitle(window, "Ray tracing mode");
					globalRenderType = RENDER_RAYTRACE;
				}
			break;}

			case GLFW_KEY_ESCAPE: {
				glfwSetWindowShouldClose(window, GL_TRUE);
			break;}

			case GLFW_KEY_I: {
				char fileName[1024];
				sprintf(fileName, "output%d.png", int(1000.0 * PCG32::rand()));
				FrameBuffer.save(fileName);
			break;}

			case GLFW_KEY_F: {
				if (!globalRecording) {
					char fileName[1024];
					sprintf(fileName, "output%d.gif", int(1000.0 * PCG32::rand()));
					printf("Saving \"%s\"...\n", fileName);
					GifBegin(&globalGIFfile, fileName, globalWidth, globalHeight, globalGIFdelay);
					globalRecording = true;
					printf("(Recording started)\n");
				} else {
					GifEnd(&globalGIFfile);
					globalRecording = false;
					printf("(Recording done)\n");
				}
			break;}

			case GLFW_KEY_W: {
				globalEye += SCLFACT * globalViewDir;
				globalLookat += SCLFACT * globalViewDir;
			break;}

			case GLFW_KEY_S: {
				globalEye -= SCLFACT * globalViewDir;
				globalLookat -= SCLFACT * globalViewDir;
			break;}

			case GLFW_KEY_Q: {
				globalEye += SCLFACT * globalUp;
				globalLookat += SCLFACT * globalUp;
			break;}

			case GLFW_KEY_Z: {
				globalEye -= SCLFACT * globalUp;
				globalLookat -= SCLFACT * globalUp;
			break;}

			case GLFW_KEY_A: {
				globalEye -= SCLFACT * globalRight;
				globalLookat -= SCLFACT * globalRight;
			break;}

			case GLFW_KEY_D: {
				globalEye += SCLFACT * globalRight;
				globalLookat += SCLFACT * globalRight;
			break;}

			default: break;
		}
	}
}



// mouse button events (you do not need to modify it unless you want to)
void mouseButtonFunc(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS) {
			mouseLeftPressed = true;
		} else if (action == GLFW_RELEASE) {
			mouseLeftPressed = false;
			if (globalRenderType == RENDER_RAYTRACE) {
				AccumulationBuffer.clear();
				sampleCount = 0;
			}
		}
	}
}



// mouse button events (you do not need to modify it unless you want to)
void cursorPosFunc(GLFWwindow* window, double mouse_x, double mouse_y) {
	if (mouseLeftPressed) {
		const float xfact = -ANGFACT * float(mouse_y - m_mouseY);
		const float yfact = -ANGFACT * float(mouse_x - m_mouseX);
		float3 v = globalViewDir;

		// local function in C++...
		struct {
			float3 operator()(float theta, const float3& v, const float3& w) {
				const float c = cosf(theta);
				const float s = sinf(theta);

				const float3 v0 = dot(v, w) * w;
				const float3 v1 = v - v0;
				const float3 v2 = cross(w, v1);

				return v0 + c * v1 + s * v2;
			}
		} rotateVector;

		v = rotateVector(xfact * DegToRad, v, globalRight);
		v = rotateVector(yfact * DegToRad, v, globalUp);
		globalViewDir = v;
		globalLookat = globalEye + globalViewDir;
		globalRight = cross(globalViewDir, globalUp);

		m_mouseX = mouse_x;
		m_mouseY = mouse_y;

		if (globalRenderType == RENDER_RAYTRACE) {
			AccumulationBuffer.clear();
			sampleCount = 0;
		}
	} else {
		m_mouseX = mouse_x;
		m_mouseY = mouse_y;
	}
}




class PointLightSource {
public:
	float3 position, wattage;
};



class Ray {
public:
	float3 o, d;
	Ray() : o(), d(float3(0.0f, 0.0f, 1.0f)) {}
	Ray(const float3& o, const float3& d) : o(o), d(d) {}
};



// uber material
// "type" will tell the actual type
// ====== implement it in A2, if you want ======
enum enumMaterialType {
	MAT_LAMBERTIAN,
	MAT_METAL,
	MAT_GLASS
};
class Material {
public:
	std::string name;

	enumMaterialType type = MAT_LAMBERTIAN;
	float eta = 1.0f;
	float glossiness = 1.0f;

	float3 Ka = float3(0.0f);
	float3 Kd = float3(0.9f);
	float3 Ks = float3(0.0f);
	float Ns = 0.0;

	// support 8-bit texture
	bool isTextured = false;
	unsigned char* texture = nullptr;
	int textureWidth = 0;
	int textureHeight = 0;

	bool isBumpMapped = false; // added for final project
	unsigned char* bumpmap = nullptr;
	int bumpMapWidth = 0;
	int bumpMapHeight = 0;



	Material() {};
	virtual ~Material() {};

	void setReflectance(const float3& c) {
		if (type == MAT_LAMBERTIAN) {
			Kd = c;
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}
	}

	float3 fetchTexture(const float2& tex) const {
		// repeating
		int x = int(tex.x * textureWidth) % textureWidth;
		int y = int(tex.y * textureHeight) % textureHeight;
		if (x < 0) x += textureWidth;
		if (y < 0) y += textureHeight;

		int pix = (x + y * textureWidth) * 3;
		const unsigned char r = texture[pix + 0];
		const unsigned char g = texture[pix + 1];
		const unsigned char b = texture[pix + 2];
		return float3(r, g, b) / 255.0f;
	}

	// added for final project, bumpmap
	float fetchBumpMap(const float2& tex) const {
		// repeating
		int x = int(tex.x * bumpMapWidth) % bumpMapWidth;
		int y = int(tex.y * bumpMapHeight) % bumpMapHeight;
		if (x < 0) x += bumpMapWidth;
		if (y < 0) y += bumpMapHeight;

		int pix = x + y * bumpMapWidth;
		const unsigned char h = bumpmap[pix];
		return float(h) / 255.0f;
	}


	float3 BRDF(const float3& wi, const float3& wo, const float3& n) const {
		float3 brdfValue = float3(0.0f);
		if (type == MAT_LAMBERTIAN) {
			// BRDF
			brdfValue = Kd / PI;
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}
		return brdfValue;
	};

	float PDF(const float3& wGiven, const float3& wSample) const {
		// probability density function for a given direction and a given sample
		// it has to be consistent with the sampler
		float pdfValue = 0.0f;
		if (type == MAT_LAMBERTIAN) {
			// empty
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}
		return pdfValue;
	}

	float3 sampler(const float3& wGiven, float& pdfValue) const {
		// sample a vector and record its probability density as pdfValue
		float3 smp = float3(0.0f);
		if (type == MAT_LAMBERTIAN) {
			// empty
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}

		pdfValue = PDF(wGiven, smp);
		return smp;
	}
};


class HitInfo {
public:
	float t; // distance
	float3 P; // location
	float3 N; // shading normal vector
	float2 T; // texture coordinate
	const Material* material; // const pointer to the material of the intersected object
	float3 dNdu; // derivative of the normal w.r.t. texture u-coordinate
    float3 dNdv; // derivative of the normal w.r.t. texture v-coordinate
};



// axis-aligned bounding box
class AABB {
private:
	float3 minp, maxp, size;

public:
	float3 get_minp() { return minp; };
	float3 get_maxp() { return maxp; };
	float3 get_size() { return size; };


	AABB() {
		minp = float3(FLT_MAX);
		maxp = float3(-FLT_MAX);
		size = float3(0.0f);
	}

	void reset() {
		minp = float3(FLT_MAX);
		maxp = float3(-FLT_MAX);
		size = float3(0.0f);
	}

	int getLargestAxis() const {
		if ((size.x > size.y) && (size.x > size.z)) {
			return 0;
		} else if (size.y > size.z) {
			return 1;
		} else {
			return 2;
		}
	}

	void fit(const float3& v) {
		if (minp.x > v.x) minp.x = v.x;
		if (minp.y > v.y) minp.y = v.y;
		if (minp.z > v.z) minp.z = v.z;

		if (maxp.x < v.x) maxp.x = v.x;
		if (maxp.y < v.y) maxp.y = v.y;
		if (maxp.z < v.z) maxp.z = v.z;

		size = maxp - minp;
	}

	float area() const {
		return (2.0f * (size.x * size.y + size.y * size.z + size.z * size.x));
	}


	bool intersect(HitInfo& minHit, const Ray& ray) const {
		// set minHit.t as the distance to the intersection point
		// return true/false if the ray hits or not
		float tx1 = (minp.x - ray.o.x) / ray.d.x;
		float ty1 = (minp.y - ray.o.y) / ray.d.y;
		float tz1 = (minp.z - ray.o.z) / ray.d.z;

		float tx2 = (maxp.x - ray.o.x) / ray.d.x;
		float ty2 = (maxp.y - ray.o.y) / ray.d.y;
		float tz2 = (maxp.z - ray.o.z) / ray.d.z;

		if (tx1 > tx2) {
			const float temp = tx1;
			tx1 = tx2;
			tx2 = temp;
		}

		if (ty1 > ty2) {
			const float temp = ty1;
			ty1 = ty2;
			ty2 = temp;
		}

		if (tz1 > tz2) {
			const float temp = tz1;
			tz1 = tz2;
			tz2 = temp;
		}

		float t1 = tx1; if (t1 < ty1) t1 = ty1; if (t1 < tz1) t1 = tz1;
		float t2 = tx2; if (t2 > ty2) t2 = ty2; if (t2 > tz2) t2 = tz2;

		if (t1 > t2) return false;
		if ((t1 < 0.0) && (t2 < 0.0)) return false;

		minHit.t = t1;
		return true;
	}
};




// triangle
struct Triangle {
	float3 positions[3];
	float3 normals[3];
	float2 texcoords[3];
	int idMaterial = 0;
	AABB bbox;
	float3 center;
};



// triangle mesh
static float3 shade(const HitInfo& hit, const float3& viewDir, const int level = 0);
class TriangleMesh {
public:
	std::vector<Triangle> triangles;
	std::vector<Material> materials;
	AABB bbox;

	void transform(const float4x4& m) {
		// ====== implement it if you want =====
		// matrix transformation of an object	
		// m is a matrix that transforms an object
		// implement proper transformation for positions and normals
		// (hint: you will need to have float4 versions of p and n)
		for (unsigned int i = 0; i < this->triangles.size(); i++) {
			for (int k = 0; k <= 2; k++) {
				const float3 &p = this->triangles[i].positions[k];
				const float3 &n = this->triangles[i].normals[k];
				// not doing anything right now
			}
		}
	}

	void rasterizeTriangle(const Triangle& tri, const float4x4& plm) const {
		// ====== implement it in A1 ======
		// rasterization of a triangle
		// "plm" should be a matrix that contains perspective projection and the camera matrix
		// you do not need to implement clipping
		// you may call the "shade" function to get the pixel value
		// (you may ignore viewDir for now)		

		// 3D coords of the triangle in world space
		float3 pos1 = tri.positions[0];
		float3 pos2 = tri.positions[1];
		float3 pos3 = tri.positions[2];
		
		float4 coord_world_1 = {pos1.x, pos1.y, pos1.z, 1.0f};
		float4 coord_world_2 = {pos2.x, pos2.y, pos2.z, 1.0f};
		float4 coord_world_3 = {pos3.x, pos3.y, pos3.z, 1.0f};

		// apply plm to each points
		float4 m1 = mul(plm, coord_world_1);
		float4 m2 = mul(plm, coord_world_2);
		float4 m3 = mul(plm, coord_world_3);

		// homogenous coordinates
		float w_homo_1 = m1.w;
		float w_homo_2 = m2.w;
		float w_homo_3 = m3.w;

		// normalize to NDC space, perspective transformation
		for (int i = 0; i < 4; i++) {
			m1[i] = m1[i] / m1.w;
			m2[i] = m2[i] / m2.w;
			m3[i] = m3[i] / m3.w;
		}

		// remap from NDC space to screen space
		float i1 = (m1.x + 1.0f) / 2.0f * globalWidth;
		float j1 = (m1.y + 1.0f) / 2.0f * globalHeight;

		float i2 = (m2.x + 1.0f) / 2.0f * globalWidth;
		float j2 = (m2.y + 1.0f) / 2.0f * globalHeight;

		float i3 = (m3.x + 1.0f) / 2.0f * globalWidth;
		float j3 = (m3.y + 1.0f) / 2.0f * globalHeight;

		// set the corresponding pixels in the framebuffer
		// fix: crush when exceeds boundary
		if (FrameBuffer.valid(i1, j1) && FrameBuffer.valid(i2, j2)
			&& FrameBuffer.valid(i3, j3)) {
			FrameBuffer.pixel(i1, j1) = float3(1.0f);
			FrameBuffer.pixel(i2, j2) = float3(1.0f);
			FrameBuffer.pixel(i3, j3) = float3(1.0f);
		} 

		float2 coord_screen_1 = {i1, j1};
		float2 coord_screen_2 = {i2, j2};
		float2 coord_screen_3 = {i3, j3};

		// use bounding box to speed up
		float xmin = std::min({coord_screen_1.x, coord_screen_2.x, coord_screen_3.x});
		float ymin = std::min({coord_screen_1.y, coord_screen_2.y, coord_screen_3.y});
		float xmax = std::max({coord_screen_1.x, coord_screen_2.x, coord_screen_3.x});
		float ymax = std::max({coord_screen_1.y, coord_screen_2.y, coord_screen_3.y});

		float step = 0.5f; // smoother
		for (float i = xmin; i <= xmax; i += step) {
			for (float j = ymin; j <= ymax; j += step) {
		// for (float i = 0; i <= globalWidth; i += step) {
		// 	for (float j = 0; j <= globalHeight; j += step) {
				float2 p = {i + 0.5f, j + 0.5f}; 

				float sign0 = edgeFunction(coord_screen_1, coord_screen_2, p);
				float sign1 = edgeFunction(coord_screen_2, coord_screen_3, p);
				float sign2 = edgeFunction(coord_screen_3, coord_screen_1, p);

				if(FrameBuffer.valid(i, j) 
					&& sign0 >= 0.0f && sign1 >= 0.0f && sign2 >= 0.0f) {
					// triangle height
					// float height_01 = distancePointToLine(coords1, coords2, coords3);
					// float height_12 = distancePointToLine(coords2, coords3, coords1);
					// float height_20 = distancePointToLine(coords1, coords3, coords2);

					// float dis_p_01 = distancePointToLine(coords1, coords2, p1);
					// float dis_p_12 = distancePointToLine(coords2, coords3, p1);
					// float dis_p_20 = distancePointToLine(coords1, coords3, p1);
					
					// float phi_01 = dis_p_01 / height_01;
					// float phi_12 = dis_p_12 / height_12;
					// float phi_20 = dis_p_20 / height_20;

					float area = triangleArea(coord_screen_1, coord_screen_2, coord_screen_3);
					float area1 = triangleArea(coord_screen_1, coord_screen_2, p);
					float area2 = triangleArea(coord_screen_2, coord_screen_3, p);
					float area3 = triangleArea(coord_screen_1, coord_screen_3, p);
					
					float phi_01 = area1 / area;
					float phi_12 = area2 / area;
					float phi_20 = area3 / area;

					float depth = phi_12 * m1.z / m1.w + phi_20 * m2.z / m2.w + phi_01 * m3.z / m3.w;
					
					float P_x = phi_12 * (tri.texcoords[0].x / w_homo_1) +
								phi_20 * (tri.texcoords[1].x / w_homo_2) +
								phi_01 * (tri.texcoords[2].x / w_homo_3);

					float P_y = phi_12 * (tri.texcoords[0].y / w_homo_1) +
								phi_20 * (tri.texcoords[1].y / w_homo_2) +
								phi_01 * (tri.texcoords[2].y / w_homo_3);

					float W = phi_12 * (1.0f / w_homo_1) + phi_20 * (1.0f  / w_homo_2) + phi_01 * (1.0f  / w_homo_3);

					float2 uv = {P_x / W, P_y / W};

					float3 norm = {phi_01 * tri.normals[2].x + phi_12 * tri.normals[0].x + phi_20 * tri.normals[1].x,
					phi_01 * tri.normals[2].y + phi_12 * tri.normals[0].y + phi_20 * tri.normals[1].y,
					phi_01 * tri.normals[2].z + phi_12 * tri.normals[0].z + phi_20 * tri.normals[1].z};

					float3 P_interpolated = {phi_01 * tri.positions[2].x + phi_12 * tri.positions[0].x + phi_20 * tri.positions[1].x,
					phi_01 * tri.positions[2].y + phi_12 * tri.positions[0].y + phi_20 * tri.positions[1].y,
					phi_01 * tri.positions[2].z + phi_12 * tri.positions[0].z + phi_20 * tri.positions[1].z};
	
					if (depth <= FrameBuffer.depth(i, j)) {
						FrameBuffer.pixel(i, j) = materials[tri.idMaterial].Kd;
						HitInfo *hit = new HitInfo();
						hit->T = uv;
						hit->N = norm;
						hit->P = P_interpolated;
						hit->material = &materials[tri.idMaterial];
						FrameBuffer.pixel(i, j) = shade(*hit, float3(0.0f));
						FrameBuffer.depth(i, j) = depth;
					}
				} 
			}
		}
	
	}

	float triangleArea(const float2 A, const float2 B, const float2 C) const {
		return 0.5 * abs(A.x*(B.y - C.y) + B.x*(C.y - A.y) + C.x*(A.y - B.y));
	}

	float edgeFunction(float2 v0, float2 v1, float2 p) const {
		return (v0.y - v1.y) * (p.x - v0.x) + (v1.x - v0.x) * (p.y - v0.y);
	}

	bool raytraceTriangle(HitInfo& result, const Ray& ray, const Triangle& tri, float tMin, float tMax) const {
		// ====== implement it in A2 ======
		// ray-triangle intersection
		// fill in "result" when there is an intersection
		// return true/false if there is an intersection or not
		// vertices of a triangle
		float3 a = tri.positions[0];
		float3 b = tri.positions[1];
		float3 c = tri.positions[2];

		// normals of a triangle
		float3 n_a = tri.normals[0];
		float3 n_b = tri.normals[1];
		float3 n_c = tri.normals[2];

		// texture coordinates of a triangle
		float2 tc_a = tri.texcoords[0];
		float2 tc_b = tri.texcoords[1];
		float2 tc_c = tri.texcoords[2];

		float3 o = ray.o;
		float3 d = ray.d;

		// coefficient matrix
		float3x3 D;
		D[0] = {a.x - b.x, a.y - b.y, a.z - b.z};
		D[1] = {a.x - c.x, a.y - c.y, a.z - c.z};
		D[2] = {d.x, d.y, d.z};

		// coefficient matrix for beta
		float3x3 D_beta;
		D_beta[0] = {a.x - o.x, a.y - o.y, a.z - o.z};
		D_beta[1] = {a.x - c.x, a.y - c.y, a.z - c.z};
		D_beta[2] = {d.x, d.y, d.z};

		// coefficient matrix for gamma
		float3x3 D_gamma;
		D_gamma[0] = {a.x - b.x, a.y - b.y, a.z - b.z};
		D_gamma[1] = {a.x - o.x, a.y - o.y, a.z - o.z};
		D_gamma[2] = {d.x, d.y, d.z};

		// coefficient matrix for t
		float3x3 D_t;
		D_t[0] = {a.x - b.x, a.y - b.y, a.z - b.z};
		D_t[1] = {a.x - c.x, a.y - c.y, a.z - c.z};
		D_t[2] = {a.x - o.x, a.y - o.y, a.z - o.z};

		// calculate the determinants
		float det_D = dot(cross(D[0], D[1]), D[2]);
		float det_beta = dot(cross(D_beta[0], D_beta[1]), D_beta[2]);
		float det_gamma = dot(cross(D_gamma[0], D_gamma[1]), D_gamma[2]);
		float det_t = dot(cross(D_t[0], D_t[1]), D_t[2]);

		// solve the equation using Cramer's rule
		float beta = det_beta / det_D;
		float gamma = det_gamma / det_D;
		float t = det_t / det_D;
		float alpha = 1 - beta - gamma; // alpha + beta + gamma = 1
		
		// interpolate the normal vector
		float3 normalVector = {alpha * n_a.x + beta * n_b.x + gamma * n_c.x,
							alpha * n_a.y + beta * n_b.y + gamma * n_c.y,
							alpha * n_a.z + beta * n_b.z + gamma * n_c.z};

		// // interpolate the textcoord
		float2 textCoord = {alpha * tc_a.x + beta * tc_b.x + gamma * tc_c.x,
							alpha * tc_a.y + beta * tc_b.y + gamma * tc_c.y};


		// // Compute the vectors from the current vertex to the two other vertices
		// float3 edge1 = a - b;
		// float3 edge2 = c - b;

		// // Compute the corresponding vector in texture space
		// float2 deltaUV1 = tc_a - tc_b;
		// float2 deltaUV2 = tc_c - tc_b;

		// float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

		// // Calculate tangent and bitangent
		// float3 tangent_a, bitangent_a;

		// tangent_a.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
		// tangent_a.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
		// tangent_a.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);

		// bitangent_a.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
		// bitangent_a.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
		// bitangent_a.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);

		// tangent_a = normalize(tangent_a);
		// bitangent_a = normalize(bitangent_a);


		// // Compute the vectors from the current vertex to the two other vertices
		// float3 edge1_b = a - b;
		// float3 edge2_b = c - b;

		// // Compute the corresponding vector in texture space
		// float2 deltaUV1_b = tc_a - tc_b;
		// float2 deltaUV2_b = tc_c - tc_b;

		// float f_b = 1.0f / (deltaUV1_b.x * deltaUV2_b.y - deltaUV2_b.x * deltaUV1_b.y);

		// // Calculate tangent and bitangent
		// float3 tangent_b, bitangent_b;

		// tangent_b.x = f_b * (deltaUV2_b.y * edge1_b.x - deltaUV1_b.y * edge2_b.x);
		// tangent_b.y = f_b * (deltaUV2_b.y * edge1_b.y - deltaUV1_b.y * edge2_b.y);
		// tangent_b.z = f_b * (deltaUV2_b.y * edge1_b.z - deltaUV1_b.y * edge2_b.z);

		// bitangent_b.x = f_b * (-deltaUV2_b.x * edge1_b.x + deltaUV1_b.x * edge2_b.x);
		// bitangent_b.y = f_b * (-deltaUV2_b.x * edge1_b.y + deltaUV1_b.x * edge2_b.y);
		// bitangent_b.z = f_b * (-deltaUV2_b.x * edge1_b.z + deltaUV1_b.x * edge2_b.z);

		// tangent_b = normalize(tangent_b);
		// bitangent_b = normalize(bitangent_b);


		// // Compute the vectors from the current vertex to the two other vertices
		// float3 edge1_c = a - c;
		// float3 edge2_c = b - c;

		// // Compute the corresponding vector in texture space
		// float2 deltaUV1_c = tc_a - tc_c;
		// float2 deltaUV2_c = tc_b - tc_c;

		// float f_c = 1.0f / (deltaUV1_c.x * deltaUV2_c.y - deltaUV2_c.x * deltaUV1_c.y);

		// // Calculate tangent and bitangent
		// float3 tangent_c, bitangent_c;

		// tangent_c.x = f_c * (deltaUV2_c.y * edge1_c.x - deltaUV1_c.y * edge2_c.x);
		// tangent_c.y = f_c * (deltaUV2_c.y * edge1_c.y - deltaUV1_c.y * edge2_c.y);
		// tangent_c.z = f_c * (deltaUV2_c.y * edge1_c.z - deltaUV1_c.y * edge2_c.z);

		// bitangent_c.x = f_c * (-deltaUV2_c.x * edge1_c.x + deltaUV1_c.x * edge2_c.x);
		// bitangent_c.y = f_c * (-deltaUV2_c.x * edge1_c.y + deltaUV1_c.x * edge2_c.y);
		// bitangent_c.z = f_c * (-deltaUV2_c.x * edge1_c.z + deltaUV1_c.x * edge2_c.z);

		// tangent_c = normalize(tangent_c);
		// bitangent_c = normalize(bitangent_c);

		// // interpolate the tangent vector
		// float3 tangentVector = {alpha * tangent_a.x + beta * tangent_b.x + gamma * tangent_c.x,
		// 						alpha * tangent_a.y + beta * tangent_b.y + gamma * tangent_c.y,
		// 						alpha * tangent_a.z + beta * tangent_b.z + gamma * tangent_c.z};

		// // interpolate the bitangent vector
		// float3 bitangentVector = {alpha * bitangent_a.x + beta * bitangent_b.x + gamma * bitangent_c.x,
		// 						alpha * bitangent_a.y + beta * bitangent_b.y + gamma * bitangent_c.y,
		// 						alpha * bitangent_a.z + beta * bitangent_b.z + gamma * bitangent_c.z};

		// accept the solution if constrains are met
		if (t > 0 && t <= tMax && t >= tMin && 
			beta > 0.0f && beta < 1.0f && gamma > 0.0f && gamma < 1.0f && 
			alpha > 0.0f && alpha < 1.0f) {
				result.t = t;
				result.P = o + d * t;	
				result.N = normalize(normalVector); // normalize it!!!
				result.T = textCoord;
				result.material = &materials[tri.idMaterial];
				// result.dNdu = normalize(tangentVector); // for bump mapping
				// result.dNdv = normalize(bitangentVector); // for bump mapping
				return true;
		}
		return false;
	}

	// some precalculation for bounding boxes (you do not need to change it)
	void preCalc() {
		bbox.reset();
		for (int i = 0, _n = (int)triangles.size(); i < _n; i++) {
			this->triangles[i].bbox.reset();
			this->triangles[i].bbox.fit(this->triangles[i].positions[0]);
			this->triangles[i].bbox.fit(this->triangles[i].positions[1]);
			this->triangles[i].bbox.fit(this->triangles[i].positions[2]);

			this->triangles[i].center = (this->triangles[i].positions[0] + this->triangles[i].positions[1] + this->triangles[i].positions[2]) * (1.0f / 3.0f);

			this->bbox.fit(this->triangles[i].positions[0]);
			this->bbox.fit(this->triangles[i].positions[1]);
			this->bbox.fit(this->triangles[i].positions[2]);
		}
	}


	// load .obj file (you do not need to modify it unless you want to change something)
	bool load(const char* filename, const float4x4& ctm = linalg::identity) {
		int nVertices = 0;
		float* vertices;
		float* normals;
		float* texcoords;
		int nIndices;
		int* indices;
		int* matid = nullptr;

		printf("Loading \"%s\"...\n", filename);
		ParseOBJ(filename, nVertices, &vertices, &normals, &texcoords, nIndices, &indices, &matid);
		if (nVertices == 0) return false;
		this->triangles.resize(nIndices / 3);

		if (matid != nullptr) {
			for (unsigned int i = 0; i < materials.size(); i++) {
				// convert .mlt data into BSDF definitions
				// you may change the followings in the final project if you want
				materials[i].type = MAT_LAMBERTIAN;
				if (materials[i].Ns == 100.0f) {
					materials[i].type = MAT_METAL;
				}
				if (materials[i].name.compare(0, 5, "glass", 0, 5) == 0) {
					materials[i].type = MAT_GLASS;
					materials[i].eta = 1.5f;
				}
			}
		} else {
			// use default Lambertian
			this->materials.resize(1);
		}

		for (unsigned int i = 0; i < this->triangles.size(); i++) {
			const int v0 = indices[i * 3 + 0];
			const int v1 = indices[i * 3 + 1];
			const int v2 = indices[i * 3 + 2];

			this->triangles[i].positions[0] = float3(vertices[v0 * 3 + 0], vertices[v0 * 3 + 1], vertices[v0 * 3 + 2]);
			this->triangles[i].positions[1] = float3(vertices[v1 * 3 + 0], vertices[v1 * 3 + 1], vertices[v1 * 3 + 2]);
			this->triangles[i].positions[2] = float3(vertices[v2 * 3 + 0], vertices[v2 * 3 + 1], vertices[v2 * 3 + 2]);

			if (normals != nullptr) {
				this->triangles[i].normals[0] = float3(normals[v0 * 3 + 0], normals[v0 * 3 + 1], normals[v0 * 3 + 2]);
				this->triangles[i].normals[1] = float3(normals[v1 * 3 + 0], normals[v1 * 3 + 1], normals[v1 * 3 + 2]);
				this->triangles[i].normals[2] = float3(normals[v2 * 3 + 0], normals[v2 * 3 + 1], normals[v2 * 3 + 2]);
			} else {
				// no normal data, calculate the normal for a polygon
				const float3 e0 = this->triangles[i].positions[1] - this->triangles[i].positions[0];
				const float3 e1 = this->triangles[i].positions[2] - this->triangles[i].positions[0];
				const float3 n = normalize(cross(e0, e1));

				this->triangles[i].normals[0] = n;
				this->triangles[i].normals[1] = n;
				this->triangles[i].normals[2] = n;
			}

			// material id
			this->triangles[i].idMaterial = 0;
			if (matid != nullptr) {
				// read texture coordinates
				if ((texcoords != nullptr) && materials[matid[i]].isTextured) {
					this->triangles[i].texcoords[0] = float2(texcoords[v0 * 2 + 0], texcoords[v0 * 2 + 1]);
					this->triangles[i].texcoords[1] = float2(texcoords[v1 * 2 + 0], texcoords[v1 * 2 + 1]);
					this->triangles[i].texcoords[2] = float2(texcoords[v2 * 2 + 0], texcoords[v2 * 2 + 1]);
				} else {
					this->triangles[i].texcoords[0] = float2(0.0f);
					this->triangles[i].texcoords[1] = float2(0.0f);
					this->triangles[i].texcoords[2] = float2(0.0f);
				}
				this->triangles[i].idMaterial = matid[i];
			} else {
				this->triangles[i].texcoords[0] = float2(0.0f);
				this->triangles[i].texcoords[1] = float2(0.0f);
				this->triangles[i].texcoords[2] = float2(0.0f);
			}
		}
		printf("Loaded \"%s\" with %d triangles.\n", filename, int(triangles.size()));

		delete[] vertices;
		delete[] normals;
		delete[] texcoords;
		delete[] indices;
		delete[] matid;

		return true;
	}

	~TriangleMesh() {
		materials.clear();
		triangles.clear();
	}


	bool bruteforceIntersect(HitInfo& result, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) {
		// bruteforce ray tracing (for debugging)
		bool hit = false;
		HitInfo tempMinHit;
		result.t = FLT_MAX;

		for (int i = 0; i < triangles.size(); ++i) {
			// debug0618
			// printf("aaaaa");
			if (raytraceTriangle(tempMinHit, ray, triangles[i], tMin, tMax)) {
				// printf("bbbb");
				if (tempMinHit.t < result.t) {
					hit = true;
					result = tempMinHit;
				}
			}
		}

		return hit;
	}

	void createSingleTriangle() {
		triangles.resize(1);
		materials.resize(1);

		triangles[0].idMaterial = 0;

		triangles[0].positions[0] = float3(-0.5f, -0.5f, 0.0f);
		triangles[0].positions[1] = float3(0.5f, -0.5f, 0.0f);
		triangles[0].positions[2] = float3(0.0f, 0.5f, 0.0f);

		const float3 e0 = this->triangles[0].positions[1] - this->triangles[0].positions[0];
		const float3 e1 = this->triangles[0].positions[2] - this->triangles[0].positions[0];
		const float3 n = normalize(cross(e0, e1));

		triangles[0].normals[0] = n;
		triangles[0].normals[1] = n;
		triangles[0].normals[2] = n;

		triangles[0].texcoords[0] = float2(0.0f, 0.0f);
		triangles[0].texcoords[1] = float2(0.0f, 1.0f);
		triangles[0].texcoords[2] = float2(1.0f, 0.0f);
	}


private:
	// === you do not need to modify the followings in this class ===
	void loadTexture(const char* fname, const int i) {
		int comp;
		materials[i].texture = stbi_load(fname, &materials[i].textureWidth, &materials[i].textureHeight, &comp, 3);
		if (!materials[i].texture) {
			std::cerr << "Unable to load texture: " << fname << std::endl;
			return;
		}
	}

	// added for final project
	void loadBumpMap(const char* fname, const int i) {
		int comp;
		materials[i].bumpmap = stbi_load(fname, &materials[i].bumpMapWidth, &materials[i].bumpMapHeight, &comp, 3);
		if (!materials[i].bumpmap) {
			std::cerr << "Unable to load bumpmap: " << fname << std::endl;
			return;
		} else {
			printf("bump map loaded!\n");
		}
	}

	std::string GetBaseDir(const std::string& filepath) {
		if (filepath.find_last_of("/\\") != std::string::npos) return filepath.substr(0, filepath.find_last_of("/\\"));
		return "";
	}
	std::string base_dir;

	void LoadMTL(const std::string fileName) {
		FILE* fp = fopen(fileName.c_str(), "r");

		Material mtl;
		mtl.texture = nullptr;
		char line[81];
		while (fgets(line, 80, fp) != nullptr) {
			float r, g, b, s;
			std::string lineStr;
			lineStr = line;
			int i = int(materials.size());

			if (lineStr.compare(0, 6, "newmtl", 0, 6) == 0) {
				lineStr.erase(0, 7);
				mtl.name = lineStr;
				mtl.isTextured = false;
			} else if (lineStr.compare(0, 2, "Ns", 0, 2) == 0) { // changed order for Blender exported mtl file
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f\n", &s);
				mtl.Ns = s;
				mtl.texture = nullptr;
			} else if (lineStr.compare(0, 2, "Ka", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Ka = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Kd", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Kd = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Ks", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Ks = float3(r, g, b);
				materials.push_back(mtl);
			} else if (lineStr.compare(0, 6, "map_Kd", 0, 6) == 0) {
				lineStr.erase(0, 7);
				lineStr.erase(lineStr.size() - 1, 1);
				materials[i - 1].isTextured = true;
				loadTexture((base_dir + lineStr).c_str(), i - 1);
			} else if (lineStr.compare(0, 7, "bumpmap", 0, 7) == 0) { // added for final project
				lineStr.erase(0, 8);
				lineStr.erase(lineStr.size() - 1, 1);
				materials[i - 1].isBumpMapped = true;
				loadBumpMap((base_dir + lineStr).c_str(), i - 1);
			}
		}

		fclose(fp);
	}

	void ParseOBJ(const char* fileName, int& nVertices, float** vertices, float** normals, float** texcoords, int& nIndices, int** indices, int** materialids) {
		// local function in C++...
		struct {
			void operator()(char* word, int* vindex, int* tindex, int* nindex) {
				const char* null = " ";
				char* ptr;
				const char* tp;
				const char* np;

				// by default, the texture and normal pointers are set to the null string
				tp = null;
				np = null;

				// replace slashes with null characters and cause tp and np to point
				// to character immediately following the first or second slash
				for (ptr = word; *ptr != '\0'; ptr++) {
					if (*ptr == '/') {
						if (tp == null) {
							tp = ptr + 1;
						} else {
							np = ptr + 1;
						}

						*ptr = '\0';
					}
				}

				*vindex = atoi(word);
				*tindex = atoi(tp);
				*nindex = atoi(np);
			}
		} get_indices;

		base_dir = GetBaseDir(fileName);
		#ifdef _WIN32
			base_dir += "\\";
		#else
			base_dir += "/";
		#endif

		FILE* fp = fopen(fileName, "r");
		int nv = 0, nn = 0, nf = 0, nt = 0;
		char line[81];
		if (!fp) {
			printf("Cannot open \"%s\" for reading\n", fileName);
			return;
		}

		while (fgets(line, 80, fp) != NULL) {
			std::string lineStr;
			lineStr = line;

			if (lineStr.compare(0, 6, "mtllib", 0, 6) == 0) {
				lineStr.erase(0, 7);
				lineStr.erase(lineStr.size() - 1, 1);
				LoadMTL(base_dir + lineStr);
			}

			if (line[0] == 'v') {
				if (line[1] == 'n') {
					nn++;
				} else if (line[1] == 't') {
					nt++;
				} else {
					nv++;
				}
			} else if (line[0] == 'f') {
				nf++;
			}
		}
		fseek(fp, 0, 0);

		float* n = new float[3 * (nn > nf ? nn : nf)];
		float* v = new float[3 * nv];
		float* t = new float[2 * nt];

		int* vInd = new int[3 * nf];
		int* nInd = new int[3 * nf];
		int* tInd = new int[3 * nf];
		int* mInd = new int[nf];

		int nvertices = 0;
		int nnormals = 0;
		int ntexcoords = 0;
		int nindices = 0;
		int ntriangles = 0;
		bool noNormals = false;
		bool noTexCoords = false;
		bool noMaterials = true;
		int cmaterial = 0;

		while (fgets(line, 80, fp) != NULL) {
			std::string lineStr;
			lineStr = line;

			if (line[0] == 'v') {
				if (line[1] == 'n') {
					float x, y, z;
					sscanf(&line[2], "%f %f %f\n", &x, &y, &z);
					float l = sqrt(x * x + y * y + z * z);
					x = x / l;
					y = y / l;
					z = z / l;
					n[nnormals] = x;
					nnormals++;
					n[nnormals] = y;
					nnormals++;
					n[nnormals] = z;
					nnormals++;
				} else if (line[1] == 't') {
					float u, v;
					sscanf(&line[2], "%f %f\n", &u, &v);
					t[ntexcoords] = u;
					ntexcoords++;
					t[ntexcoords] = v;
					ntexcoords++;
				} else {
					float x, y, z;
					sscanf(&line[1], "%f %f %f\n", &x, &y, &z);
					v[nvertices] = x;
					nvertices++;
					v[nvertices] = y;
					nvertices++;
					v[nvertices] = z;
					nvertices++;
				}
			}
			if (lineStr.compare(0, 6, "usemtl", 0, 6) == 0) {
				lineStr.erase(0, 7);
				if (materials.size() != 0) {
					for (unsigned int i = 0; i < materials.size(); i++) {
						if (lineStr.compare(materials[i].name) == 0) {
							cmaterial = i;
							noMaterials = false;
							break;
						}
					}
				}

			} else if (line[0] == 'f') {
				char s1[32], s2[32], s3[32];
				int vI, tI, nI;
				sscanf(&line[1], "%s %s %s\n", s1, s2, s3);

				mInd[ntriangles] = cmaterial;

				// indices for first vertex
				get_indices(s1, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				// indices for second vertex
				get_indices(s2, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				// indices for third vertex
				get_indices(s3, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				ntriangles++;
			}
		}

		*vertices = new float[ntriangles * 9];
		if (!noNormals) {
			*normals = new float[ntriangles * 9];
		} else {
			*normals = 0;
		}

		if (!noTexCoords) {
			*texcoords = new float[ntriangles * 6];
		} else {
			*texcoords = 0;
		}

		if (!noMaterials) {
			*materialids = new int[ntriangles];
		} else {
			*materialids = 0;
		}

		*indices = new int[ntriangles * 3];
		nVertices = ntriangles * 3;
		nIndices = ntriangles * 3;

		for (int i = 0; i < ntriangles; i++) {
			if (!noMaterials) {
				(*materialids)[i] = mInd[i];
			}

			(*indices)[3 * i] = 3 * i;
			(*indices)[3 * i + 1] = 3 * i + 1;
			(*indices)[3 * i + 2] = 3 * i + 2;

			(*vertices)[9 * i] = v[3 * vInd[3 * i]];
			(*vertices)[9 * i + 1] = v[3 * vInd[3 * i] + 1];
			(*vertices)[9 * i + 2] = v[3 * vInd[3 * i] + 2];

			(*vertices)[9 * i + 3] = v[3 * vInd[3 * i + 1]];
			(*vertices)[9 * i + 4] = v[3 * vInd[3 * i + 1] + 1];
			(*vertices)[9 * i + 5] = v[3 * vInd[3 * i + 1] + 2];

			(*vertices)[9 * i + 6] = v[3 * vInd[3 * i + 2]];
			(*vertices)[9 * i + 7] = v[3 * vInd[3 * i + 2] + 1];
			(*vertices)[9 * i + 8] = v[3 * vInd[3 * i + 2] + 2];

			if (!noNormals) {
				(*normals)[9 * i] = n[3 * nInd[3 * i]];
				(*normals)[9 * i + 1] = n[3 * nInd[3 * i] + 1];
				(*normals)[9 * i + 2] = n[3 * nInd[3 * i] + 2];

				(*normals)[9 * i + 3] = n[3 * nInd[3 * i + 1]];
				(*normals)[9 * i + 4] = n[3 * nInd[3 * i + 1] + 1];
				(*normals)[9 * i + 5] = n[3 * nInd[3 * i + 1] + 2];

				(*normals)[9 * i + 6] = n[3 * nInd[3 * i + 2]];
				(*normals)[9 * i + 7] = n[3 * nInd[3 * i + 2] + 1];
				(*normals)[9 * i + 8] = n[3 * nInd[3 * i + 2] + 2];
			}

			if (!noTexCoords) {
				(*texcoords)[6 * i] = t[2 * tInd[3 * i]];
				(*texcoords)[6 * i + 1] = t[2 * tInd[3 * i] + 1];

				(*texcoords)[6 * i + 2] = t[2 * tInd[3 * i + 1]];
				(*texcoords)[6 * i + 3] = t[2 * tInd[3 * i + 1] + 1];

				(*texcoords)[6 * i + 4] = t[2 * tInd[3 * i + 2]];
				(*texcoords)[6 * i + 5] = t[2 * tInd[3 * i + 2] + 1];
			}

		}
		fclose(fp);

		delete[] n;
		delete[] v;
		delete[] t;
		delete[] nInd;
		delete[] vInd;
		delete[] tInd;
		delete[] mInd;
	}
};



// BVH node (for A2 extra)
class BVHNode {
public:
	bool isLeaf;
	int idLeft, idRight;
	int triListNum;
	int* triList;
	AABB bbox;
};


// ====== implement it in A2 extra ======
// fill in the missing parts
class BVH {
public:
	const TriangleMesh* triangleMesh = nullptr;
	BVHNode* node = nullptr;

	const float costBBox = 1.0f;
	const float costTri = 1.0f;

	int leafNum = 0;
	int nodeNum = 0;

	BVH() {}
	void build(const TriangleMesh* mesh);

	bool intersect(HitInfo& result, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) const {
		bool hit = false;
		HitInfo tempMinHit;
		result.t = FLT_MAX;

		// bvh
		if (this->node[0].bbox.intersect(tempMinHit, ray)) {
			hit = traverse(result, ray, 0, tMin, tMax);
		}
		if (result.t != FLT_MAX) hit = true;

		return hit;
	}
	bool traverse(HitInfo& result, const Ray& ray, int node_id, float tMin, float tMax) const;

private:
	void sortAxis(int* obj_index, const char axis, const int li, const int ri) const;
	int splitBVH(int* obj_index, const int obj_num, const AABB& bbox);

};


// sort bounding boxes (in case you want to build SAH-BVH)
void BVH::sortAxis(int* obj_index, const char axis, const int li, const int ri) const {
	int i, j;
	float pivot;
	int temp;

	i = li;
	j = ri;

	pivot = triangleMesh->triangles[obj_index[(li + ri) / 2]].center[axis];

	while (true) {
		while (triangleMesh->triangles[obj_index[i]].center[axis] < pivot) {
			++i;
		}

		while (triangleMesh->triangles[obj_index[j]].center[axis] > pivot) {
			--j;
		}

		if (i >= j) break;

		temp = obj_index[i];
		obj_index[i] = obj_index[j];
		obj_index[j] = temp;

		++i;
		--j;
	}

	if (li < (i - 1)) sortAxis(obj_index, axis, li, i - 1);
	if ((j + 1) < ri) sortAxis(obj_index, axis, j + 1, ri);
}


//#define SAHBVH // use this in once you have SAH-BVH
int BVH::splitBVH(int* obj_index, const int obj_num, const AABB& bbox) {
	// ====== exntend it in A2 extra ======
#ifndef SAHBVH
	int bestAxis, bestIndex;
	AABB bboxL, bboxR, bestbboxL, bestbboxR;
	int* sorted_obj_index  = new int[obj_num];

	// split along the largest axis
	bestAxis = bbox.getLargestAxis();

	// sorting along the axis
	this->sortAxis(obj_index, bestAxis, 0, obj_num - 1);
	for (int i = 0; i < obj_num; ++i) {
		sorted_obj_index[i] = obj_index[i];
	}

	// split in the middle
	bestIndex = obj_num / 2 - 1;

	bboxL.reset();
	for (int i = 0; i <= bestIndex; ++i) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];
		bboxL.fit(tri.positions[0]);
		bboxL.fit(tri.positions[1]);
		bboxL.fit(tri.positions[2]);
	}

	bboxR.reset();
	for (int i = bestIndex + 1; i < obj_num; ++i) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];
		bboxR.fit(tri.positions[0]);
		bboxR.fit(tri.positions[1]);
		bboxR.fit(tri.positions[2]);
	}

	bestbboxL = bboxL;
	bestbboxR = bboxR;
#else
	// implelement SAH-BVH here
#endif

	if (obj_num <= 4) {
		delete[] sorted_obj_index;

		this->nodeNum++;
		this->node[this->nodeNum - 1].bbox = bbox;
		this->node[this->nodeNum - 1].isLeaf = true;
		this->node[this->nodeNum - 1].triListNum = obj_num;
		this->node[this->nodeNum - 1].triList = new int[obj_num];
		for (int i = 0; i < obj_num; i++) {
			this->node[this->nodeNum - 1].triList[i] = obj_index[i];
		}
		int temp_id;
		temp_id = this->nodeNum - 1;
		this->leafNum++;

		return temp_id;
	} else {
		// split obj_index into two 
		int* obj_indexL = new int[bestIndex + 1];
		int* obj_indexR = new int[obj_num - (bestIndex + 1)];
		for (int i = 0; i <= bestIndex; ++i) {
			obj_indexL[i] = sorted_obj_index[i];
		}
		for (int i = bestIndex + 1; i < obj_num; ++i) {
			obj_indexR[i - (bestIndex + 1)] = sorted_obj_index[i];
		}
		delete[] sorted_obj_index;
		int obj_numL = bestIndex + 1;
		int obj_numR = obj_num - (bestIndex + 1);

		// recursive call to build a tree
		this->nodeNum++;
		int temp_id;
		temp_id = this->nodeNum - 1;
		this->node[temp_id].bbox = bbox;
		this->node[temp_id].isLeaf = false;
		this->node[temp_id].idLeft = splitBVH(obj_indexL, obj_numL, bestbboxL);
		this->node[temp_id].idRight = splitBVH(obj_indexR, obj_numR, bestbboxR);

		delete[] obj_indexL;
		delete[] obj_indexR;

		return temp_id;
	}
}


// you may keep this part as-is
void BVH::build(const TriangleMesh* mesh) {
	triangleMesh = mesh;

	// construct the bounding volume hierarchy
	const int obj_num = (int)(triangleMesh->triangles.size());
	int* obj_index = new int[obj_num];
	for (int i = 0; i < obj_num; ++i) {
		obj_index[i] = i;
	}
	this->nodeNum = 0;
	this->node = new BVHNode[obj_num * 2];
	this->leafNum = 0;

	// calculate a scene bounding box
	AABB bbox;
	for (int i = 0; i < obj_num; i++) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];

		bbox.fit(tri.positions[0]);
		bbox.fit(tri.positions[1]);
		bbox.fit(tri.positions[2]);
	}

	// ---------- buliding BVH ----------
	printf("Building BVH...\n");
	splitBVH(obj_index, obj_num, bbox);
	printf("Done.\n");

	delete[] obj_index;
}


// you may keep this part as-is
bool BVH::traverse(HitInfo& minHit, const Ray& ray, int node_id, float tMin, float tMax) const {
	bool hit = false;
	HitInfo tempMinHit, tempMinHitL, tempMinHitR;
	bool hit1, hit2;

	if (this->node[node_id].isLeaf) {
		for (int i = 0; i < (this->node[node_id].triListNum); ++i) {
			if (triangleMesh->raytraceTriangle(tempMinHit, ray, triangleMesh->triangles[this->node[node_id].triList[i]], tMin, tMax)) {
				hit = true;
				if (tempMinHit.t < minHit.t) minHit = tempMinHit;
			}
		}
	} else {
		hit1 = this->node[this->node[node_id].idLeft].bbox.intersect(tempMinHitL, ray);
		hit2 = this->node[this->node[node_id].idRight].bbox.intersect(tempMinHitR, ray);

		hit1 = hit1 && (tempMinHitL.t < minHit.t);
		hit2 = hit2 && (tempMinHitR.t < minHit.t);

		if (hit1 && hit2) {
			if (tempMinHitL.t < tempMinHitR.t) {
				hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
				hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
			} else {
				hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
				hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
			}
		} else if (hit1) {
			hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
		} else if (hit2) {
			hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
		}
	}

	return hit;
}




// ====== implement it in A3 ======
// fill in the missing parts
class Particle {
public:
	float3 position = float3(0.0f);
	float3 velocity = float3(0.0f);
	float3 prevPosition = position;

	float3 c = float3(0.0f); // Task 3, center of the bounding circle
	float3 accumulatedForce; // Task 4
	float radius; // Task 5, radius of a sphere

	void reset() {
		position = float3(PCG32::rand(), PCG32::rand(), PCG32::rand()) - float(0.5f);
		velocity = 2.0f * float3((PCG32::rand() - 0.5f), 0.0f, (PCG32::rand() - 0.5f));
		prevPosition = position;
		position += velocity * deltaT;
	}

	void step() {
		// === fill in this part in A3 ===
		// update the particle position and velocity here

		// Task 1
		// Verlet Integration
		// float3 nextPosition = 2.0f * position - prevPosition + deltaT * deltaT * globalGravity;
		// prevPosition = position;
		// position = nextPosition;

		// Task 2
		// Use position-based method to deal with collition
		// float3 nextPosition = 2.0f * position - prevPosition + deltaT * deltaT * globalGravity;
		// if (position.x > -0.5f && nextPosition.x < -0.5f) {
		// 	position.x = -position.x + nextPosition.x - 0.5f;
		// 	nextPosition.x = -0.5f;
		// }
		// if (position.x < 0.5f && nextPosition.x > 0.5f) {
		// 	position.x = -position.x + nextPosition.x + 0.5f;
		// 	nextPosition.x = 0.5f;
		// }

		// if (position.y > -0.5f && nextPosition.y < -0.5f) {
		// 	position.y = -position.y + nextPosition.y - 0.5f;
		// 	nextPosition.y = -0.5f;
		// }
		// if (position.y < 0.5f && nextPosition.y > 0.5f) {
		// 	position.y = -position.y + nextPosition.y + 0.5f;
		// 	nextPosition.y = 0.5f;
		// }

		// if (position.z > -0.5f && nextPosition.z < -0.5f) {
		// 	position.z = -position.z + nextPosition.z - 0.5f;
		// 	nextPosition.z = -0.5f;
		// } 
		// if (position.z < 0.5f && nextPosition.z > 0.5f) {
		// 	position.z = -position.z + nextPosition.z + 0.5f;
		// 	nextPosition.z = 0.5f;
		// }

		// prevPosition = position;
		// position = nextPosition;

		// Task 3
		// Position-based method
		// float r = 0.53f;
		// float3 nextPosition = 2.0f * position - prevPosition + deltaT * deltaT * globalGravity;
		// nextPosition = c + r * ((nextPosition - c) / distance(nextPosition, c)); 
		// prevPosition = position;
		// position = nextPosition;

		// Task 4
		// Position-based method
		// float3 nextPosition = 2.0f * position - prevPosition + deltaT * deltaT * accumulatedForce;
		// prevPosition = position;
		// position = nextPosition;

		// Task 5
		// Position-based method
		// float r = 0.53f;
		// float3 nextPosition = 2.0f * position - prevPosition + deltaT * deltaT * globalGravity;
		// nextPosition = c + r * ((nextPosition - c) / distance(nextPosition, c)); 
		// prevPosition = position;
		// position = nextPosition;
	}

	
};



class ParticleSystem {
public:
	std::vector<Particle> particles;
	TriangleMesh particlesMesh;
	TriangleMesh sphere;
	const char* sphereMeshFilePath = 0;
	float sphereSize = 0.0f;
	ParticleSystem() {};

	void updateMesh() {
		// you can optionally update the other mesh information (e.g., bounding box, BVH - which is tricky)
		if (sphereSize > 0) {
			const int n = int(sphere.triangles.size());
			for (int i = 0; i < globalNumParticles; i++) {
				for (int j = 0; j < n; j++) {
					particlesMesh.triangles[i * n + j].positions[0] = sphere.triangles[j].positions[0] + particles[i].position;
					particlesMesh.triangles[i * n + j].positions[1] = sphere.triangles[j].positions[1] + particles[i].position;
					particlesMesh.triangles[i * n + j].positions[2] = sphere.triangles[j].positions[2] + particles[i].position;
					particlesMesh.triangles[i * n + j].normals[0] = sphere.triangles[j].normals[0];
					particlesMesh.triangles[i * n + j].normals[1] = sphere.triangles[j].normals[1];
					particlesMesh.triangles[i * n + j].normals[2] = sphere.triangles[j].normals[2];
				}
			}
		} else {
			const float particleSize = 0.005f;
			for (int i = 0; i < globalNumParticles; i++) {
				// facing toward the camera
				particlesMesh.triangles[i].positions[0] = particles[i].position;
				particlesMesh.triangles[i].positions[1] = particles[i].position + particleSize * globalUp;
				particlesMesh.triangles[i].positions[2] = particles[i].position + particleSize * globalRight;
				particlesMesh.triangles[i].normals[0] = -globalViewDir;
				particlesMesh.triangles[i].normals[1] = -globalViewDir;
				particlesMesh.triangles[i].normals[2] = -globalViewDir;
			}
		}
	}

	void initialize() {
		particles.resize(globalNumParticles);
		particlesMesh.materials.resize(1);
		for (int i = 0; i < globalNumParticles; i++) {
			particles[i].reset();
		}

		if (sphereMeshFilePath) {
			if (sphere.load(sphereMeshFilePath)) {
				particlesMesh.triangles.resize(sphere.triangles.size() * globalNumParticles);
				sphere.preCalc();
				sphereSize = sphere.bbox.get_size().x * 0.5f;
			} else {
				particlesMesh.triangles.resize(globalNumParticles);
			}
		} else {
			particlesMesh.triangles.resize(globalNumParticles);
		}

		// Added for task 5	
		for (int i = 0; i < globalNumParticles; i++) {
			particles[i].radius = sphereSize;
		}

		updateMesh();
	}

	// For task 5, comment in this fuction
	// void step() {
	// 	// Iterate over all pairs of particles (spheres)
	// 	for (int i = 0; i < globalNumParticles; i++) {
	// 		for (int j = i + 1; j < globalNumParticles; j++) {
	// 			if (collision(particles[i], particles[j])) {
	// 				// If a collision occurs, resolve it using position-based method
	// 				float dp = 2.0f * sphereSize - distance(particles[i].position, particles[j].position);
	// 				// Euclidean distance between the centers of two spheres
	// 				float distance1 = distance(particles[i].position, particles[j].position);
	// 				// Direction that particle[i] should be pushed towards
	// 				float3 direction = (particles[i].position - particles[j].position) / distance1;
	// 				// New positions of sepheres. Two spheres are pushed in opposite directions from each other.
	// 				particles[i].position = particles[i].position + direction * dp / 2.0f;
	// 				particles[j].position = particles[j].position + (-direction) * dp / 2.0f;
	// 			}
	// 		}
	// 	}

	// 	for (int i = 0; i < globalNumParticles; i++) {
	// 		particles[i].step();
	// 	}
	// 	updateMesh();
	// }

	// Added for task 5, detect if there is a collition between two spheres
	bool collision(const Particle& sphereA, const Particle& sphereB) {
		float distance_1 = distance(sphereB.position, sphereA.position);
		if (distance_1 <= sphereA.radius + sphereB.radius) {
			return true;
		}
		return false; 
	}

	// For task 4, comment in this fuction
	void step() {
		// Iterate over all pairs of particles
		for (int i = 0; i < globalNumParticles; i++) {
			float3 accumulatedForce = float3(0.0f);
			for (int j = 0; j < globalNumParticles; j++) {
				if (i != j) {
					// Euclidean distance between two particles
					float distance_1 = distance(particles[j].position, particles[i].position);
					// Square of Euclidean distance between two particles
        			float distance_2 = distance2(particles[j].position, particles[i].position);
					// Cube of Euclidean distance between two particles
					float distance_3 = distance_1 * distance_2;
					// Calculate and add the gravity force
        			accumulatedForce += (18e-3f) / distance_3 * (particles[j].position - particles[i].position);
				} 
			}
			particles[i].accumulatedForce = accumulatedForce;
		}

		for (int i = 0; i < globalNumParticles; i++) {
			particles[i].step();
		}
		updateMesh();
	}

	
};
static ParticleSystem globalParticleSystem;


// Added for A2 Task5 
static Image envMap;
void loadEnvironmentMap() {
    // envMap.load("../media/uffizi_probe.hdr");
	envMap.load("../media/stpeters_probe.hdr");
}

// scene definition
class Scene {
public:
	std::vector<TriangleMesh*> objects;
	std::vector<PointLightSource*> pointLightSources;
	std::vector<BVH> bvhs;

	void addObject(TriangleMesh* pObj) {
		objects.push_back(pObj);
	}
	void addLight(PointLightSource* pObj) {
		pointLightSources.push_back(pObj);
	}

	void preCalc() {
		bvhs.resize(objects.size());
		for (int i = 0; i < objects.size(); i++) {
			objects[i]->preCalc();
			bvhs[i].build(objects[i]);
		}
	}

	// ray-scene intersection
	bool intersect(HitInfo& minHit, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) const {
		bool hit = false;
		HitInfo tempMinHit;
		minHit.t = FLT_MAX;

		for (int i = 0, i_n = (int)objects.size(); i < i_n; i++) {
			//if (objects[i]->bruteforceIntersect(tempMinHit, ray, tMin, tMax)) { // for debugging
			if (bvhs[i].intersect(tempMinHit, ray, tMin, tMax)) {
				if (tempMinHit.t < minHit.t) {
					hit = true;
					minHit = tempMinHit;
				}
			}
		}
		return hit;
	}

	// camera -> screen matrix (given to you for A1)
	float4x4 perspectiveMatrix(float fovy, float aspect, float zNear, float zFar) const {
		float4x4 m;
		const float f = 1.0f / (tan(fovy * DegToRad / 2.0f));
		m[0] = { f / aspect, 0.0f, 0.0f, 0.0f };
		m[1] = { 0.0f, f, 0.0f, 0.0f };
		m[2] = { 0.0f, 0.0f, (zFar + zNear) / (zNear - zFar), -1.0f };
		m[3] = { 0.0f, 0.0f, (2.0f * zFar * zNear) / (zNear - zFar), 0.0f };

		return m;
	}

	// model -> camera matrix (given to you for A1)
	float4x4 lookatMatrix(const float3& _eye, const float3& _center, const float3& _up) const {
		// transformation to the camera coordinate
		float4x4 m;
		const float3 f = normalize(_center - _eye);
		const float3 upp = normalize(_up);
		const float3 s = normalize(cross(f, upp));
		const float3 u = cross(s, f);

		m[0] = { s.x, s.y, s.z, 0.0f };
		m[1] = { u.x, u.y, u.z, 0.0f };
		m[2] = { -f.x, -f.y, -f.z, 0.0f };
		m[3] = { 0.0f, 0.0f, 0.0f, 1.0f };
		m = transpose(m);

		// translation according to the camera location
		const float4x4 t = float4x4{ {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}, { -_eye.x, -_eye.y, -_eye.z, 1.0f} };

		m = mul(m, t);
		return m;
	}

	// rasterizer
	void Rasterize() const {
		// printf("into rasterizer, ");

		auto start = std::chrono::high_resolution_clock::now();
		// ====== implement it in A1 ======
		// fill in plm by a proper matrix
		const float4x4 pm = perspectiveMatrix(globalFOV, globalAspectRatio, globalDepthMin, globalDepthMax);
		const float4x4 lm = lookatMatrix(globalEye, globalLookat, globalUp);
		const float4x4 plm = mul(pm, lm);

		FrameBuffer.clear();
		for (int n = 0, n_n = (int)objects.size(); n < n_n; n++) {
			// debug1
			for (int k = 0, k_n = (int)objects[n]->triangles.size(); k < k_n; k++) {
					objects[n]->rasterizeTriangle(objects[n]->triangles[k], plm);
			}
		}


		auto end = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

		// printf("time cost = %d \n", duration);
	}

	// eye ray generation (given to you for A2)
	// Ray eyeRay(int x, int y) const {
	// 	// compute the camera coordinate system 
	// 	const float3 wDir = normalize(float3(-globalViewDir));
	// 	const float3 uDir = normalize(cross(globalUp, wDir));
	// 	const float3 vDir = cross(wDir, uDir);

	// 	// compute the pixel location in the world coordinate system using the camera coordinate system
	// 	// trace a ray through the center of each pixel
	// 	const float imPlaneUPos = (x + 0.5f) / float(globalWidth) - 0.5f;
	// 	const float imPlaneVPos = (y + 0.5f) / float(globalHeight) - 0.5f;

	// 	const float3 pixelPos = globalEye + float(globalAspectRatio * globalFilmSize * imPlaneUPos) * uDir + float(globalFilmSize * imPlaneVPos) * vDir - globalDistanceToFilm * wDir;

	// 	return Ray(globalEye, normalize(pixelPos - globalEye));
	// }
	// // srand(static_cast<unsigned int>(time(0)));
	
	// Function that generates a random float between 0 and 1
	float randFloat() const {
		// Generate a random integer and divide it by the maximum random integer value
		// to get a random float between 0 and 1
		return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
	}

	// Function to generate a random point inside a unit disk (used for simulating lens effect)
	float3 randomInLens() const {
		float3 point;
		do {
			// Generate a random point in a square [-1, 1] x [-1, 1]
			point.x = (2.0f * randFloat() - 1.0f);
			point.y = (2.0f * randFloat() - 1.0f);
		// If the point is outside the unit disk, generate a new point
		} while (length(point) > 1.0f);
		return point;
	}

	// eyeRay for depth of field
	Ray eyeRay(int x, int y, float lensRadius, float focalDistance) const {
		// Compute the camera coordinate system (u, v, w)
		float3 wDir = normalize(float3(-globalViewDir));
		float3 uDir = normalize(cross(globalUp, wDir));
		float3 vDir = cross(wDir, uDir);

		// Normalize the pixel coordinates to [-0.5, 0.5] range
		float imPlaneUPos = (x + 0.5f) / float(globalWidth) - 0.5f;
		float imPlaneVPos = (y + 0.5f) / float(globalHeight) - 0.5f;

		// Compute the original point on the image plane
		float3 imagePoint = globalEye + float(globalAspectRatio * globalFilmSize * imPlaneUPos) * uDir 
								+ float(globalFilmSize * imPlaneVPos) * vDir 
								- globalDistanceToFilm * wDir;

		// Compute the point on the focal plane
		float3 focalPoint = globalEye + (imagePoint - globalEye) * (focalDistance / globalDistanceToFilm);

		// Compute a random point on the lens
		float3 lensPoint = globalEye + lensRadius * randomInLens() * uDir 
								+ lensRadius * randomInLens() * vDir;

		// Generate a ray from the lens point to the focal point
		return Ray(lensPoint, normalize(focalPoint - lensPoint));
	}


	// added for A2 Task 5, fetch color form env map
	float3 fetchEnv(const float3& viewDir) const {
		float r = (1.0f / PI) * acos(viewDir.z) / sqrt(viewDir.x * viewDir.x + viewDir.y * viewDir.y);
		// r is in [-1, 1], so convert it into [0,1] firstly
		float x = (viewDir.x * r + 1.0f) / 2.0f;
		float y = (viewDir.y * r + 1.0f) / 2.0f;
		// calculate the cooridinate on the env map
		int u = static_cast<int>(x * envMap.width + 0.5f);
		int v = static_cast<int>(y * envMap.height + 0.5f);
		return envMap.pixel(u, v);
	}

	// ray tracing (you probably don't need to change it in A2)
	// void Raytrace() const {
	// 	FrameBuffer.clear();

	// 	// loop over all pixels in the image
	// 	for (int j = 0; j < globalHeight; ++j) {
	// 		for (int i = 0; i < globalWidth; ++i) {
	// 			const Ray ray = eyeRay(i, j);
	// 			HitInfo hitInfo;
	// 			if (intersect(hitInfo, ray)) {
	// 				// printf("hit!! ");
	// 				FrameBuffer.pixel(i, j) = shade(hitInfo, -ray.d);
	// 			} else {
	// 				// printf("hit!! ");
	// 				FrameBuffer.pixel(i, j) = float3(0.0f);
	// 				// added for A2 Task 5 to set background image, comment out this for task1-4
	// 				// FrameBuffer.pixel(i, j) = fetchEnv(ray.d); 
	// 			}
	// 		}

	// 		// show intermediate process
	// 		if (globalShowRaytraceProgress) {
	// 			constexpr int scanlineNum = 64;
	// 			if ((j % scanlineNum) == (scanlineNum - 1)) {
	// 				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, globalWidth, globalHeight, GL_RGB, GL_FLOAT, &FrameBuffer.pixels[0]);
	// 				glRecti(1, 1, -1, -1);
	// 				glfwSwapBuffers(globalGLFWindow);
	// 				printf("Rendering Progress: %.3f%%\r", j / float(globalHeight - 1) * 100.0f);
	// 				fflush(stdout);
	// 			}
	// 		}
	// 	}
	// }

	// Glossy reflection
	// void Raytrace() const {
	// 	FrameBuffer.clear();
	// 	const int samplesPerPixel = 10; 
	// 	// loop over all pixels in the image
	// 	for (int j = 0; j < globalHeight; ++j) {
	// 		for (int i = 0; i < globalWidth; ++i) {
	// 			float3 pixelColor(0.0f);
	// 			for (int s = 0; s < samplesPerPixel; ++s) {
	// 				const Ray ray = eyeRay(i, j);
	// 				HitInfo hitInfo;
	// 				if (intersect(hitInfo, ray)) {
	// 					pixelColor += shade(hitInfo, -ray.d);
	// 				} else {
	// 					pixelColor += fetchEnv(ray.d); 
	// 				}
	// 			}	
	// 			FrameBuffer.pixel(i, j) = pixelColor / static_cast<float>(samplesPerPixel);
	// 		}
	// 	}
	// }

	// Depth of field
	void Raytrace() const {
		FrameBuffer.clear();
		const float apertureRadius = 0.031f; 
		const float focalDistance = 2.37f; // yuexiao jinchu yue qingqi
		const int samplesPerPixel = 900;  

		// DoF, smoothed, less noise
		// Loop over multiple samples for each pixel
		for (int j = 0; j < globalHeight; ++j) {
			for (int i = 0; i < globalWidth; ++i) {
				float3 pixelColor(0.0f);
				for (int s = 0; s < samplesPerPixel; ++s) {
					const Ray ray = eyeRay(i, j, apertureRadius, focalDistance);
					HitInfo hitInfo;
					if (intersect(hitInfo, ray)) {
						pixelColor += shade(hitInfo, -ray.d);
					} else {
						pixelColor += float3(0.0f);  // Or use background color here
					}
				}
				// Compute the average color for this pixel by dividing the accumulated color by the number of samples
				FrameBuffer.pixel(i, j) = pixelColor / static_cast<float>(samplesPerPixel);
			}
		}
	}

	// Anti-aliasing
	// void Raytrace() const {
	// 	FrameBuffer.clear();
	// 	const int numSamples = 8; // change this to the number of samples you want
	// 	// loop over all pixels in the image
	// 	for (int j = 0; j < globalHeight; ++j) {
	// 		for (int i = 0; i < globalWidth; ++i) {
	// 			float3 color(0.0f);
	// 			// super sampling loop
	// 			for (int p = 0; p < numSamples; ++p) {
	// 				for (int q = 0; q < numSamples; ++q) {
	// 					float u = i + (p + (float) rand() / (RAND_MAX)) / numSamples;
    //                 	float v = j + (q + (float) rand() / (RAND_MAX)) / numSamples;
	// 					const Ray ray = eyeRay(u, v);
	// 					HitInfo hitInfo;
	// 					if (intersect(hitInfo, ray)) {
	// 						color += shade(hitInfo, -ray.d);
	// 					} else {
	// 						color += float3(0.0f);
	// 					}
	// 				}
	// 			}
	// 			// average the color
	// 			color /= numSamples * numSamples;
	// 			FrameBuffer.pixel(i, j) = color;
	// 		}
	// 	}
	// }

};
static Scene globalScene;



// ====== implement it in A2 ======
// fill in the missing parts

// Final project: for soft shadow
float traceShadowRay(const HitInfo& hit, const PointLightSource& light, float radius, int sampleCount) {
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(-1.0, 1.0);
    int hitCount = 0;
    float scale = radius * 2;
    // Generate multiple shadow rays for each hit point
	#pragma omp parallel for reduction(+:hitCount)
    for (int i = 0; i < sampleCount; i++) {
        // Jitter the position of the light to simulate an area light
        float3 lightPos = light.position;
        // lightPos.x += distribution(generator) * scale;
    	// lightPos.y += distribution(generator) * scale;
    	// lightPos.z += distribution(generator) * scale;

		// Here rand() / (float)RAND_MAX generates a random floating point number between 0 and 1. 
		// We subtract 0.5 and multiply by radius * 2 to get a random number between -radius and radius.
		lightPos.x += ((rand() / (float)RAND_MAX) - 0.5f) * radius * 2;
		lightPos.y += ((rand() / (float)RAND_MAX) - 0.5f) * radius * 2;
		lightPos.z += ((rand() / (float)RAND_MAX) - 0.5f) * radius * 2;

        // Construct a ray from the hit point towards the jittered light position
        float3 lightDir = normalize(lightPos - hit.P);
        Ray shadowRay{hit.P, lightDir};

        // Check if the shadow ray hits any objects before it reaches the light
        HitInfo shadowHit;

        float lightDist = length(lightPos - hit.P);

        if (globalScene.intersect(shadowHit, shadowRay, Epsilon, lightDist)) {
            // The shadow ray hit an object before it reached the light
            // Count this as a shadow hit
            hitCount++;
        }
    }
    
    // Calculate the proportion of shadow rays that were blocked
    float shadowRatio = hitCount / (float)sampleCount;
    
    // Return the shadow ratio
    // This will be a value between 0 (completely lit) and 1 (completely shadowed)
    // We can use this value to interpolate between the lit and shadowed color of the hit point
    return shadowRatio;
}



double genRandom(double lower, double upper) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(lower, upper);

	return dis(gen);
}

float3 randomPointInSphere() {
    // Using rejection sampling to generate a random point within a unit sphere
    float3 p;
    static const float3 center(1.0f, 1.0f, 1.0f);
    do {
        p = 2.0f * float3(genRandom(0.0, 1.0), genRandom(0.0, 1.0), genRandom(0.0, 1.0)) - center;
    } while (p.x * p.x + p.y * p.y + p.z * p.z >= 1.0f);
    return p;
}


float randFloat()  {
	// Generate a random integer and divide it by the maximum random integer value
	// to get a random float between 0 and 1
	return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}


static float3 shade(const HitInfo& hit, const float3& viewDir, const int level) {
	if (hit.material->type == MAT_LAMBERTIAN) {
		// you may want to add shadow ray tracing here in A2
		float3 L = float3(0.0f);
		float3 brdf, irradiance;
		// loop over all of the point light sources
		for (int i = 0; i < globalScene.pointLightSources.size(); i++) {
			// final project: bump map implementation
			float3 hitNormal = hit.N;
			if (hit.material->isBumpMapped) {
				// printf("into isBumpMapped ");
				float u = hit.T.x;
				float v = hit.T.y;
				int x = static_cast<int>(u * (hit.material->bumpMapWidth - 1));
				int y = static_cast<int>(v * (hit.material->bumpMapHeight - 1));
				// Fetch bump map height at the nearest corner
				float h = hit.material->fetchBumpMap(float2(u, v));
				// Compute gradients in the u and v directions
				float dhdu = hit.material->fetchBumpMap(float2(float(x+1)/hit.material->bumpMapWidth, v)) - h;
				float dhdv = hit.material->fetchBumpMap(float2(u, float(y+1)/hit.material->bumpMapHeight)) - h;
				// Compute the perturbed normal
				float3 bumpedNormal = normalize(hit.N + dhdu * hit.dNdu + dhdv * hit.dNdv);
				// Replace the normal with the bumped normal for the remaining of the shading computation
				hitNormal = bumpedNormal;
			}

			float3 l = globalScene.pointLightSources[i]->position - hit.P;
			// the inverse-squared falloff
			const float falloff = length2(l);
			// normalize the light direction
			l /= sqrtf(falloff);
			Ray shadowRay(hit.P +hitNormal * Epsilon, l);
			HitInfo shadowHit; // hit point on the object between light and shadow hitpoint
			// get the irradiance
			irradiance = float(std::max(0.0f, dot(hitNormal, l)) / (4.0 * PI * falloff)) * globalScene.pointLightSources[i]->wattage;
			brdf = hit.material->BRDF(l, viewDir,hitNormal);

			if (hit.material->isTextured) {
				// Comment out this line for texture mapping based on bilinear interpolation 
				brdf *= hit.material->fetchTexture(hit.T);

				// bilinear interpolation of the texture BEGIN
				// float u = hit.T.x;
				// float v = hit.T.y;

				// float ui = u * (hit.material->textureWidth - 1);
				// float vi = v * (hit.material->textureHeight - 1);

				// int x = static_cast<int>(ui);
				// int y = static_cast<int>(vi);

				// int xNext = std::min(x + 1, hit.material->textureWidth - 1);
				// int yNext = std::min(y + 1, hit.material->textureHeight - 1);

				// float dx = ui - x;
				// float dy = vi - y;

				// float3 color00 = hit.material->fetchTexture(float2(u, v));
				// float3 color10 = hit.material->fetchTexture(float2(float(xNext)/hit.material->textureWidth, v));
				// float3 color01 = hit.material->fetchTexture(float2(u, float(yNext)/hit.material->textureHeight));
				// float3 color11 = hit.material->fetchTexture(float2(float(xNext)/hit.material->textureWidth, float(yNext)/hit.material->textureHeight));

				// float3 color = (1 - dx) * (1 - dy) * color00 + dx * (1 - dy) * color10 +
				// 				(1 - dx) * dy * color01 + dx * dy * color11;

				// brdf *= color;
				// bilinear interpolation of the texture END
			}

			//  Comment out that line to enable illumination computation for a point light source. 
			// return brdf * PI; //debug output

			// Soft shadow BEGIN
			float radius = 0.8f;
			int sampleCount1 = 400;
			float shadowRatio = traceShadowRay(hit, *globalScene.pointLightSources[i], radius, sampleCount1);
			L += (1.0f - shadowRatio) * irradiance * brdf;
			// Soft shadow END

			// Hard shadow BEGIN
			// point is in shadow, skip
			// float length = std::sqrt(shadowRay.d.x * shadowRay.d.x + shadowRay.d.y * shadowRay.d.y + shadowRay.d.z * shadowRay.d.z);
			// if (globalScene.intersect(shadowHit, shadowRay, Epsilon, length)) {
			// 	continue;
			// }
			// L += irradiance * brdf;
			// Hard shadow END
		}
		return L;
	} else if (hit.material->type == MAT_METAL) {
		// maximum depth of ray tracing to avoid infinite loop
		const int maxDepth = 5;
		if (level >= maxDepth) {
			return float3(0.0f);
		}

		// reflected ray
		float3 r = -2.0f * dot(-viewDir, hit.N) * hit.N + (-viewDir); 

		// for glossy reflection, add random points
		r += 0.08f * randomPointInSphere() ;
   		r = normalize(r);

		// create a new ray, using small offset to avoid self-intersection
		Ray reflectedRay(hit.P + hit.N * Epsilon, r);
		HitInfo reflectedHit; 

		float length = std::sqrt(reflectedRay.d.x * reflectedRay.d.x + reflectedRay.d.y * reflectedRay.d.y + reflectedRay.d.z * reflectedRay.d.z);
		// if ray hits an object, calculate its color contribution
		if (globalScene.intersect(reflectedHit, reflectedRay, Epsilon, length)) {
			// return hit.material->Ks*shade(reflectedHit, -r, level + 1);
			return shade(reflectedHit, -r, level + 1);
		} else {
			// comment in for A2 Task 5, if not hit, get a value from the env map
			// return globalScene.fetchEnv(r); 
			return float3(0.0f);
		}
	} else if (hit.material->type == MAT_GLASS) {
		// maximum depth of ray tracing to avoid infinite loop
		const int maxDepth = 5;
		if (level >= maxDepth) {
			return float3(0.0f);
		}

		HitInfo refractedHit; 
		HitInfo reflectedHit; 

		// index of refraction
		float eta1 = 1.0f;
		float eta2 = hit.material->eta;
		// float eta2 = 1.0f;
		
		// the ray is entering the glass
		if (dot(-viewDir, hit.N) < 0) {
			// equation under the square root
			float tmp = 1.0f - (eta1 / eta2) * (eta1 / eta2) * 
					(1.0f - dot(-viewDir, hit.N) * dot(-viewDir, hit.N));

			// refracted ray
			float3 w_t = (eta1 / eta2) * (-viewDir - dot(-viewDir, hit.N) * hit.N) 
					- (sqrtf(tmp)) * hit.N;
			w_t = normalize(w_t);
			
			// create a new ray, using small offset to avoid self-intersection
			Ray refractedRay(hit.P + hit.N * Epsilon, w_t);
			float length = std::sqrt(refractedRay.d.x * refractedRay.d.x + 
									refractedRay.d.y * refractedRay.d.y + 
									refractedRay.d.z * refractedRay.d.z);
			if (globalScene.intersect(refractedHit, refractedRay, Epsilon, length)) {
				// return hit.material->Ks*shade(reflectedHit, -r, level + 1);
				return shade(refractedHit, -w_t, level + 1);
			} else {
				return float3(0.0f);
			}
			
			//  the ray is exiting glass
		} else if (dot(-viewDir, hit.N) > 0) {
			// equation under the square root
			float tmp = 1.0f - (eta2 / eta1) * (eta2 / eta1) * 
					(1.0f - dot(-viewDir, -hit.N) * dot(-viewDir, -hit.N));

			// refracted ray
			float3 w_t = (eta2 / eta1) * (-viewDir - dot(-viewDir, -hit.N) * (-hit.N)) 
					- (sqrtf(tmp)) * (-hit.N);
			w_t = normalize(w_t);

			// reflected ray
			float3 r = -2.0f * dot(-viewDir, -hit.N) * (-hit.N) + (-viewDir);
			r = normalize(r);

			// create a new ray, using small offset to avoid self-intersection
			Ray refractedRay(hit.P + (-hit.N) * Epsilon, w_t);	
			Ray reflectedRay(hit.P + hit.N * Epsilon, r);

			float length1 = std::sqrt(refractedRay.d.x * refractedRay.d.x + refractedRay.d.y * refractedRay.d.y + refractedRay.d.z * refractedRay.d.z);

			float length2 = std::sqrt(reflectedRay.d.x * reflectedRay.d.x + reflectedRay.d.y * reflectedRay.d.y + reflectedRay.d.z * reflectedRay.d.z);


			// the value under the square root is positive, calculate refraction
			if (tmp > 0.0f) {
				if (globalScene.intersect(refractedHit, refractedRay, Epsilon, length1)) {
					// return hit.material->Ks*shade(reflectedHit, -w_t, level + 1);
					return shade(refractedHit, -w_t, level + 1);
				} else {
					return float3(0.0f);
				}
			} else if (tmp < 0.0f) {
				// total internal reflection
				if (globalScene.intersect(reflectedHit, reflectedRay, Epsilon, length2)) {
					// return hit.material->Ks*shade(reflectedHit, -r, level + 1);
					return shade(reflectedHit, -r, level + 1);
				} else {
					return float3(0.0f);
				}
			}
		}
	} else {
		// something went wrong - make it apparent that it is an error
		return float3(100.0f, 0.0f, 100.0f);
	}
}


// OpenGL initialization (you will not use any OpenGL/Vulkan/DirectX... APIs to render 3D objects!)
// you probably do not need to modify this in A0 to A3.
class OpenGLInit {
public:
	OpenGLInit() {
		// initialize GLFW
		if (!glfwInit()) {
			std::cerr << "Failed to initialize GLFW." << std::endl;
			exit(-1);
		}

		// create a window
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
		globalGLFWindow = glfwCreateWindow(globalWidth, globalHeight, "Welcome!", NULL, NULL);
		if (globalGLFWindow == NULL) {
			std::cerr << "Failed to open GLFW window." << std::endl;
			glfwTerminate();
			exit(-1);
		}

		// make OpenGL context for the window
		glfwMakeContextCurrent(globalGLFWindow);

		// initialize GLEW
		glewExperimental = true;
		if (glewInit() != GLEW_OK) {
			std::cerr << "Failed to initialize GLEW." << std::endl;
			glfwTerminate();
			exit(-1);
		}

		// set callback functions for events
		glfwSetKeyCallback(globalGLFWindow, keyFunc);
		glfwSetMouseButtonCallback(globalGLFWindow, mouseButtonFunc);
		glfwSetCursorPosCallback(globalGLFWindow, cursorPosFunc);

		// create shader
		FSDraw = glCreateProgram();
		GLuint s = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(s, 1, &PFSDrawSource, 0);
		glCompileShader(s);
		glAttachShader(FSDraw, s);
		glLinkProgram(FSDraw);

		// create texture
		glActiveTexture(GL_TEXTURE0);
		glGenTextures(1, &GLFrameBufferTexture);
		glBindTexture(GL_TEXTURE_2D, GLFrameBufferTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, globalWidth, globalHeight, 0, GL_LUMINANCE, GL_FLOAT, 0);

		// initialize some OpenGL state (will not change)
		glDisable(GL_DEPTH_TEST);

		glUseProgram(FSDraw);
		glUniform1i(glGetUniformLocation(FSDraw, "input_tex"), 0);

		GLint dims[4];
		glGetIntegerv(GL_VIEWPORT, dims);
		const float BufInfo[4] = { float(dims[2]), float(dims[3]), 1.0f / float(dims[2]), 1.0f / float(dims[3]) };
		glUniform4fv(glGetUniformLocation(FSDraw, "BufInfo"), 1, BufInfo);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, GLFrameBufferTexture);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	}

	virtual ~OpenGLInit() {
		glfwTerminate();
	}
};



// main window
// you probably do not need to modify this in A0 to A3.
class Window {
public:
	// put this first to make sure that the glInit's constructor is called before the one for Window
	OpenGLInit glInit;

	Window() {}
	virtual ~Window() {}

	void(*process)() = NULL;

	void start() const {
		if (globalEnableParticles) {
			globalScene.addObject(&globalParticleSystem.particlesMesh);
		}
		globalScene.preCalc();

		// main loop
		while (glfwWindowShouldClose(globalGLFWindow) == GL_FALSE) {
			glfwPollEvents();
			// printf("globalEye     x : %f, y : %f, z : %f\n", globalEye.x, globalEye.y, globalEye.z);
			// printf("globalLookAt  x : %f, y : %f, z : %f\n\n", globalLookat.x, globalLookat.y, globalLookat.z);
			globalViewDir = normalize(globalLookat - globalEye);
			globalRight = normalize(cross(globalViewDir, globalUp));

			if (globalEnableParticles) {
				globalParticleSystem.step();
			}

			if (globalRenderType == RENDER_RASTERIZE) {
				globalScene.Rasterize();
			} else if (globalRenderType == RENDER_RAYTRACE) {
				globalScene.Raytrace();
			} else if (globalRenderType == RENDER_IMAGE) {
				if (process) process();
			}

			if (globalRecording) {
				unsigned char* buf = new unsigned char[FrameBuffer.width * FrameBuffer.height * 4];
				int k = 0;
				for (int j = FrameBuffer.height - 1; j >= 0; j--) {
					for (int i = 0; i < FrameBuffer.width; i++) {
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).x));
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).y));
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).z));
						buf[k++] = 255;
					}
				}
				GifWriteFrame(&globalGIFfile, buf, globalWidth, globalHeight, globalGIFdelay);
				delete[] buf;
			}

			// drawing the frame buffer via OpenGL (you don't need to touch this)
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, globalWidth, globalHeight, GL_RGB, GL_FLOAT, &FrameBuffer.pixels[0][0]);
			glRecti(1, 1, -1, -1);
			glfwSwapBuffers(globalGLFWindow);
			globalFrameCount++;
			PCG32::rand();
		}
	}
};


