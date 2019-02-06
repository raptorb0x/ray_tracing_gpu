__kernel void render_kernel(__global float3* output, int width, int height)
{
 const int work_item_id = get_global_id(0); /* the unique global id of the work item for the current pixel */
 int x = work_item_id % width; /* x-coordinate of the pixel */
 int y = work_item_id / width; /* y-coordinate of the pixel */
 float fx = (float)x / (float)width; 
 float fy = (float)y / (float)height; 
 output[work_item_id] = (float3)(fx, fy, 0);
}
