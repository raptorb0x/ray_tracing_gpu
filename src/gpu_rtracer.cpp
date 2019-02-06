#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <sstream>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <CL/cl.hpp>

#include "geometry.h"
using std::cin;
using std::cout;
using std::endl;
using std::cerr;
using std::ifstream;
using std::string;
using std::vector;
//using cl::cl_float4;
using cl::CommandQueue;
using cl::Kernel;
using cl::Context;
using cl::Program;
using cl::Buffer;
using cl::Platform;
using cl::Device;


const int   width    =800; //962
const int   height   =640;
const float fov      = M_PI/3.;
const int   r_depth  =6;
cl_float4* cpu_output;
CommandQueue queue;
Kernel kernel;
Context context;
Program program;
Buffer cl_output;

void pickPlatform(Platform& platform, const vector<Platform>& platforms){
	
	if (platforms.size() == 1) platform = platforms[0];
	else{
		int input = 0;
		cout << "\nChoose an OpenCL platform: ";
		cin >> input;

		// handle incorrect user input
		while (input < 1 || input > platforms.size()){
			cin.clear(); //clear errors/bad flags on cin
			cin.ignore(cin.rdbuf()->in_avail(), '\n'); // ignores exact number of chars in cin buffer
			cout << "No such option. Choose an OpenCL platform: ";
			cin >> input;
		}
		platform = platforms[input - 1];
	}
}

void pickDevice(Device& device, const vector<Device>& devices){
	
	if (devices.size() == 1) device = devices[0];
	else{
		int input = 0;
		cout << "\nChoose an OpenCL device: ";
		cin >> input;

		// handle incorrect user input
		while (input < 1 || input > devices.size()){
			cin.clear(); //clear errors/bad flags on cin
			cin.ignore(cin.rdbuf()->in_avail(), '\n'); // ignores exact number of chars in cin buffer
			cout << "No such option. Choose an OpenCL device: ";
			cin >> input;
		}
		device = devices[input - 1];
	}
}

void printErrorLog(const Program& program, const Device& device){
	
	// Get the error log and print to console
	string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
	cerr << "Build log:" << endl << buildlog << endl;

	// Print the error log to a file
	FILE *log = fopen("errorlog.txt", "w");
	fprintf(log, "%s\n", buildlog);
	cout << "Error log saved in 'errorlog.txt'" << endl;
	system("PAUSE");
	exit(1);
}

void cleanUp(){
	delete cpu_output;
}

void initOpenCL()
{
	// Get all available OpenCL platforms (e.g. AMD OpenCL, Nvidia CUDA, Intel OpenCL)
	vector<Platform> platforms;
	Platform::get(&platforms);
	cout << "Available OpenCL platforms : " << endl << endl;
	for (int i = 0; i < platforms.size(); i++)
		cout << "\t" << i + 1 << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << endl;

	// Pick one platform
	Platform platform;
	pickPlatform(platform, platforms);
	cout << "\nUsing OpenCL platform: \t" << platform.getInfo<CL_PLATFORM_NAME>() << endl;

	// Get available OpenCL devices on platform
	vector<Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

	cout << "Available OpenCL devices on this platform: " << endl << endl;
	for (int i = 0; i < devices.size(); i++){
		cout << "\t" << i + 1 << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << endl;
		cout << "\t\tMax compute units: " << devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
		cout << "\t\tMax work group size: " << devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl << endl;
	}

	// Pick one device
	Device device;
	pickDevice(device, devices);
	cout << "\nUsing OpenCL device: \t" << device.getInfo<CL_DEVICE_NAME>() << endl;
	cout << "\t\t\tMax compute units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
	cout << "\t\t\tMax work group size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;

	// Create an OpenCL context and command queue on that device.
	context = Context(device);
	queue = CommandQueue(context, device);

	// Convert the OpenCL source code to a string
	string source;
	ifstream file("opencl_kernel.cl");
	if (!file){
		cout << "\nNo OpenCL file found!" << endl << "Exiting..." << endl;
		system("PAUSE");
		exit(1);
	}
	while (!file.eof()){
		char line[256];
		file.getline(line, 255);
		source += line;
	}

	const char* kernel_source = source.c_str();

	// Create an OpenCL program by performing runtime source compilation for the chosen device
	program = Program(context, kernel_source);
	cl_int result = program.build({ device });
	if (result) cout << "Error during compilation OpenCL code!!!\n (" << result << ")" << endl;
	if (result == CL_BUILD_PROGRAM_FAILURE) printErrorLog(program, device);

	// Create a kernel (entry point in the OpenCL source program)
	kernel = Kernel(program, "render_kernel");
}

inline float clamp(float x){ return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }

// convert RGB float in range [0,1] to int in range [0, 255]
inline int toInt(float x){ return int(clamp(x) * 255 + .5); }



void render_opencl( uint8_t* pixmap) {
    /*
    #pragma omp parallel for 
    for (size_t j = 0; j<height; j++) {
        for (size_t i = 0; i<width; i++) {
            framebuffer[i+j*width] = Vec3f(j/float(height),i/float(width), 0);
        }
    }*/
// every pixel in the image has its own thread or "work item",
	// so the total amount of work items equals the number of pixels
	size_t global_work_size = width * height;
	size_t local_work_size = 64; 

	// launch the kernel
	queue.enqueueNDRangeKernel(kernel, NULL, global_work_size, local_work_size);
	queue.finish();
    // read and copy OpenCL output to CPU
	queue.enqueueReadBuffer(cl_output, CL_TRUE, 0, width * height * sizeof(cl_float3), cpu_output);

    #pragma omp parallel for
    for (size_t i = 0; i < height*width; ++i) {

        for (size_t j = 0; j<3; j++) { //j++?
            pixmap[i*4+j] = toInt(cpu_output[i].s[j]);
        }
        pixmap[4*i+3] = 255;
    }
    
}

vector<Vec3f> framebuffer(width*height);
void render_plain( uint8_t* pixmap) {
    
    #pragma omp parallel for 
    for (size_t j = 0; j<height; j++) {
        for (size_t i = 0; i<width; i++) {
            framebuffer[i+j*width] = Vec3f(j/float(height),i/float(width), 0);
        }
    }

    for (size_t i = 0; i < height*width; ++i) {
        Vec3f &c = framebuffer[i];
        float max = std::max(c[0], std::max(c[1], c[2]));
        if (max>1) c = c*(1./max);
        for (size_t j = 0; j<3; j++) { //j++?
            pixmap[i*4+j] = (uint8_t)(255 * std::max(0.f, std::min(1.f, framebuffer[i][j])));
        }
        pixmap[4*i+3] = 255;
    }
    
}


int main() {
    // allocate memory on CPU to hold image
	cpu_output = new cl_float3[width * height];

	// initialise OpenCL
	initOpenCL();

	// Create image buffer on the OpenCL device
	cl_output = Buffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(cl_float3));

    // specify OpenCL kernel arguments
    kernel.setArg(0, cl_output);
	kernel.setArg(1, width);
	kernel.setArg(2, height);

    
    
    

	
    sf::RenderWindow window(sf::VideoMode(width, height), "SFML");
    window.setFramerateLimit(60);
    sf::Texture texture;
    texture.create(width,height);
    sf::Sprite sprite(texture);
    uint8_t* pixmap = new uint8_t[width*height*4]; //in heap
    //uint8_t pixmap[width*height*4]; in stack
    int i=0;   
    clock_t current_ticks, delta_ticks;
    float fps = 0;
    char buffer[4];
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
			{
                window.close();
                cleanUp();
			}
        }

        current_ticks = clock();

        
        //render_plain( pixmap);
		render_opencl(pixmap);
        
        texture.update(pixmap);
        delta_ticks = clock() - current_ticks; //the time, in ms, that took to render the scene
        if(delta_ticks > 0)
        fps = CLOCKS_PER_SEC*1.0/ delta_ticks;
        sprintf(buffer,"%.2f", fps);
        window.setTitle(buffer); 
        window.clear();
        window.draw(sprite);
        window.display();

        // std::cout << fps << std::endl;
        i++;
    }

    

    return 0;
}

