#include "ray_tracer.h"
Window window;

// draw something in each frame
static void draw() {
    for (int j = 0; j < globalHeight; j++) {
        for (int i = 0; i < globalWidth; i++) {
            FrameBuffer.pixel(i, j) = float3(PCG32::rand()); // noise
            // FrameBuffer.pixel(i, j) = float3(0.5f * (cos((i + globalFrameCount) * 0.1f) + 1.0f)); // moving cosine
        }
    }
}

// setting up lighting
static PointLightSource light;
static PointLightSource light1;
static PointLightSource light2;
static void setupLightSource() {
    // added for multiple light source
    light1.position = float3(-2.6f, 3.0f, 4.3f);
    light1.wattage = float3(650.0f, 650.0f, 650.0f);
    globalScene.addLight(&light1);

    light.position = float3(3.0f, 3.0f, 3.0f);
    light.wattage = float3(800.0f, 800.0f, 800.0f);
    globalScene.addLight(&light);

    light2.position = float3(-0.9f, 3.0f, -0.5f);
    light2.wattage = float3(200.0f, 200.0f, 200.0f);
    globalScene.addLight(&light2);
}

// loading .obj file from the command line arguments
static TriangleMesh mesh;
static void setupScene(int argc, const char* argv[]) {
    if (argc > 1) {
        bool objLoadSucceed = mesh.load(argv[1]);
        if (!objLoadSucceed) {
            printf("Invalid .obj file.\n");
            printf("Making a single triangle instead.\n");
            mesh.createSingleTriangle();
        }
    } else {
        printf("Specify .obj file in the command line arguments. Example: ex.exe cornellbox.obj\n");
        printf("Making a single triangle instead.\n");
        mesh.createSingleTriangle();
    }
    globalScene.addObject(&mesh);
}

int main(int argc, const char* argv[]) {
    srand(static_cast<unsigned int>(time(0)));
    setupScene(argc, argv);
    setupLightSource();
    globalRenderType = RENDER_RAYTRACE;
    loadEnvironmentMap(); 
    window.start();
}
