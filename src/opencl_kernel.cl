/* OpenCL ray tracing tutorial by Sam Lapere, 2016
http://raytracey.blogspot.com */

__constant float EPSILON = 0.00003f; /* required to compensate for limited float precision */
__constant float PI = 3.14159265359f;

typedef struct Ray{
	float3 origin;
	float3 dir;
};

typedef Sphere{
	float3 pos;
	float radius;

};

bool intersect_sphere(const struct Sphere* sphere, const struct Ray* ray, float* t)
{
	float3 rayToCenter = sphere->pos - ray->origin;

	/* calculate coefficients a, b, c from quadratic equation */

	/* float a = dot(ray->dir, ray->dir); // ray direction is normalised, dotproduct simplifies to 1 */ 
	float b = dot(rayToCenter, ray->dir);
	float c = dot(rayToCenter, rayToCenter) - sphere->radius*sphere->radius;
	float disc = b * b - c; /* discriminant of quadratic formula */

	/* solve for t (distance to hitpoint along ray) */

	if (disc < 0.0f) return false;
	else *t = b - sqrt(disc);

	if (*t < 0.0f){
		*t = b + sqrt(disc);
		if (*t < 0.0f) return false; 
	}

	else return true;
}



__kernel void render_kernel(__constant Sphere* spheres, const int width, const int height, const int sphere_count, __global float3* output)
{
	unsigned int work_item_id = get_global_id(0);	/* the unique global id of the work item for the current pixel */
	unsigned int x_coord = work_item_id % width;			/* x-coordinate of the pixel */
	unsigned int y_coord = work_item_id / width;			/* y-coordinate of the pixel */

	float x = (2*(float)x_coord) / (float)width;  /* convert int in range [0 - width] to float in range [0-1] */
	float y = (float)y_coord / (float)height; /* convert int in range [0 - height] to float in range [0-1] */


	
	

	/* intersect ray with sphere */
	float t = 1e20;
	intersect_sphere(&sphere1, &camray, &t);

	/* if ray misses sphere, return background colour 
	background colour is a blue-ish gradient dependent on image height */
	if (t > 1e19 && rendermode != 1){ 
		output[work_item_id] = (float3)(fy * 0.1f, fy * 0.3f, 0.3f);
		return;
	}

	/* for more interesting lighting: compute normal 
	and cosine of angle between normal and ray direction */
	float3 hitpoint = camray.origin + camray.dir * t;
	float3 normal = normalize(hitpoint - sphere1.pos);
	float cosine_factor = dot(normal, camray.dir) * -1.0f;
	
	output[work_item_id] = sphere1.color * cosine_factor;

	/* six different rendermodes */
	if (rendermode == 1) output[work_item_id] = (float3)(fx, fy, 0); /* simple interpolated colour gradient based on pixel coordinates */
	else if (rendermode == 2) output[work_item_id] = sphere1.color;  /* raytraced sphere with plain colour */
	else if (rendermode == 3) output[work_item_id] = sphere1.color * cosine_factor; /* with cosine weighted colour */
	else if (rendermode == 4) output[work_item_id] = sphere1.color * cosine_factor * sin(80 * fy); /* with sinusoidal stripey pattern */
	else if (rendermode == 5) output[work_item_id] = sphere1.color * cosine_factor * sin(400 * fy) * sin(400 * fx); /* with grid pattern */
	else output[work_item_id] = normal * 0.5f + (float3)(0.5f, 0.5f, 0.5f); /* with normal colours */
}