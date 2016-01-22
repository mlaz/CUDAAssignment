
// Based on CUDA SDK template from NVIDIA
// sgm algorithm adapted from http://lunokhod.org/?p=1403

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <limits.h>

#include <limits>
#include <algorithm>

// includes, project
#include <cutil_inline.h>

#define MMAX_BRIGHTNESS 255

#define PENALTY1 15
#define PENALTY2 100

#define COSTS(i,j,d)              costs[(i)*disp_range+(j)*nx*disp_range+(d)]
#define ACCUMULATED_COSTS(i,j,d)  accumulated_costs[(i)*disp_range+(j)*nx*disp_range+(d)]
#define LEFT_IMAGE(i,j)           left_image[(i)+(j)*nx]
#define RIGHT_IMAGE(i,j)          right_image[(i)+(j)*nx]
#define DISP_IMAGE(i,j)           disp_image[(i)+(j)*nx]

#define MMAX(a,b) (((a)>(b))?(a):(b))
#define MMIN(a,b) (((a)<(b))?(a):(b))

/* function headers */

void determine_costs(const int *left_image, const int *right_image, int *costs,
                     const int nx, const int ny, const int disp_range);

void evaluate_path( const int *prior, const int* local,
                    int path_intensity_gradient, int *curr_cost,
                    const int nx, const int ny, const int disp_range );

void iterate_direction_dirxpos(const int dirx, const int *left_image,
                               const int* costs, int *accumulated_costs,
                               const int nx, const int ny, const int disp_range );

void iterate_direction_dirypos(const int diry, const int *left_image,
                               const int* costs, int *accumulated_costs,
                               const int nx, const int ny, const int disp_range );

void iterate_direction_dirxneg(const int dirx, const int *left_image,
                               const int* costs, int *accumulated_costs,
                               const int nx, const int ny, const int disp_range );

void iterate_direction_diryneg(const int diry, const int *left_image,
                               const int* costs, int *accumulated_costs,
                               const int nx, const int ny, const int disp_range );

void iterate_direction( const int dirx, const int diry, const int *left_image,
                        const int* costs, int *accumulated_costs,
                        const int nx, const int ny, const int disp_range ) ;

void d_iterate_direction ( dim3 block, dim3 grid, const int* d_costs,
                           const int *d_left_image, int *d_accumulated_costs,
                           const int dirx, const int diry, const int *left_image,
                           const int* costs, int *accumulated_costs,
                           const int nx, const int ny, const int disp_range );

void inplace_sum_views( int * im1, const int * im2,
                        const int nx, const int ny, const int disp_range ) ;

int find_min_index( const int *v, const int dist_range ) ;

void create_disparity_view( const int *accumulated_costs , int * disp_image, int nx, int ny) ;
void sgmHost(   const int *h_leftIm, const int *h_rightIm,
                int *h_dispIm,
                const int w, const int h, const int disp_range );

void sgmDevice( const int *h_leftIm, const int *h_rightIm,
                int *h_dispImD,
                const int w, const int h, const int disp_range );

void usage(char *command);

__device__ int d_find_min_index ( const int *v, const int disp_range );

__device__ void d_evaluate_path ( int *prior, int *local,
                                int path_intensity_gradient, int *curr_cost,
                                  int nx, int ny, int disp_range );

/* device functions and kernels */
__global__ void d_determine_costs ( int *left_image, int *right_image, int *costs,
                                  int nx, int ny, int disp_range )
{
  int row_size = ceil((float) nx / blockDim.x);
  int x = ((blockIdx.x % row_size) * blockDim.x) + threadIdx.x;
  int y = blockIdx.y;
  int d = ((blockIdx.x / row_size) * blockDim.y) + threadIdx.y;
  if ( (y < ny) && (d < disp_range) && (x < nx))
    {
      COSTS(x,y,d) = 255u;
      if (x >= d)
        COSTS(x,y,d) = abs( LEFT_IMAGE(x,y) - RIGHT_IMAGE(x-d,y) );
    }
}


__global__ void d_iterate_direction_dirxpos ( const int dirx, const int *left_image,
                                 int* costs, int *accumulated_costs,
                                 const int nx, const int ny, const int disp_range )
{
  int x = 0;
  int y = blockIdx.y;
  int d = threadIdx.z;

  if ( (y < ny) && (d < disp_range) )
    {
      ACCUMULATED_COSTS(0,y,d) += COSTS(0,y,d);
      __syncthreads();
      for (x = 1; x < nx; x++)
        {
          d_evaluate_path( &ACCUMULATED_COSTS(x-dirx,y,0),
                           &COSTS(x,y,0),
                           abs(LEFT_IMAGE(x,y)-LEFT_IMAGE(x-dirx,y)) ,
                           &ACCUMULATED_COSTS(x,y,0), nx, ny, disp_range);
        }

    }
}

__global__ void d_iterate_direction_dirypos ( const int diry, const int *left_image,
                                              int* costs, int *accumulated_costs,
                                              const int nx, const int ny, const int disp_range )
{
  int x = blockIdx.y;
  int y = 0;
  int d = threadIdx.z;

  if ( (x < nx) && (d < disp_range) )
    {
      ACCUMULATED_COSTS(x,0,d) += COSTS(x,0,d);;
      __syncthreads();
      for (y = 1; y < ny; y++)
        {
          d_evaluate_path( &ACCUMULATED_COSTS(x,y-diry,0),
                           &COSTS(x,y,0),
                           abs(LEFT_IMAGE(x,y)-LEFT_IMAGE(x,y-diry)) ,
                           &ACCUMULATED_COSTS(x,y,0), nx, ny, disp_range);
        }

    }
}

__global__ void d_iterate_direction_dirxneg ( const int dirx, const int *left_image,
                                            int* costs, int *accumulated_costs,
                                            const int nx, const int ny, const int disp_range )
{
  int x = nx-1;
  int y = blockIdx.y;
  int d = threadIdx.z;

  if ( (y < ny) && (d < disp_range) )
    {
      ACCUMULATED_COSTS(nx-1,y,d) += COSTS(nx-1,y,d);
      __syncthreads();
      for (x = nx-2; x >= 0; x--)
        {
          d_evaluate_path( &ACCUMULATED_COSTS(x-dirx,y,0),
                           &COSTS(x,y,0),
                           abs(LEFT_IMAGE(x,y)-LEFT_IMAGE(x-dirx,y)) ,
                           &ACCUMULATED_COSTS(x,y,0), nx, ny, disp_range);
        }

    }
}

__global__ void d_iterate_direction_diryneg ( const int diry, const int *left_image,
                                 int* costs, int *accumulated_costs,
                                 const int nx, const int ny, const int disp_range )
{
  int x = blockIdx.y;
  int y = ny-1;
  int d = threadIdx.z;

  if ( (x < nx) && (d < disp_range) )
    {
      ACCUMULATED_COSTS(x,ny-1,d) += COSTS(x,ny-1,d);;
      __syncthreads();
      for (y = ny-2; y >= 0; y--)
        {
          d_evaluate_path( &ACCUMULATED_COSTS(x,y-diry,0),
                           &COSTS(x,y,0),
                           abs(LEFT_IMAGE(x,y)-LEFT_IMAGE(x,y-diry)) ,
                           &ACCUMULATED_COSTS(x,y,0), nx, ny, disp_range);
        }

    }
}


__device__ void d_evaluate_path ( int *prior, int *local,
                                  int path_intensity_gradient, int *curr_cost,
                                  int nx, int ny, int disp_range )
{
  int d = threadIdx.z;
  curr_cost[d] = local[threadIdx.z];
  __syncthreads();
  int e_smooth = INT_MAX;
  for ( int d_p = 0; d_p < disp_range; d_p++ )
    {
      if ( d_p - d == 0 ) {
        // No penality
        e_smooth = MMIN(e_smooth,prior[d_p]);
      } else if ( abs(d_p - d) == 1 ) {
        // Small penality
        e_smooth = MMIN(e_smooth,prior[d_p]+PENALTY1);
      } else {
        // Large penality
        e_smooth =
          MMIN(e_smooth,prior[d_p] +
               MMAX(PENALTY1,
                    path_intensity_gradient ? PENALTY2/path_intensity_gradient : PENALTY2));
      }
    }
  __syncthreads();
  curr_cost[d] += e_smooth;

  int min = INT_MAX;
  for ( int d1 = 0; d1 < disp_range; d1++ ) {
        if (prior[d1]<min) min=prior[d1];
  }

  __syncthreads();
  curr_cost[d] -= min;
}


__global__ void d_inplace_sum_views ( int * im1, const int * im2,
                                      const int nx, const int ny, const int disp_range )
{
  int pos = (blockIdx.x * blockDim.x) + threadIdx.x;
  int size = nx * ny * disp_range;
  if ( pos < size )
    im1[pos] += im2[pos];
}

__global__ void d_create_disparity_view ( int *accumulated_costs , int * disp_image,
                                        int nx, int ny, int disp_range )
{
  int pos = (blockIdx.x * blockDim.x) + threadIdx.x;
  int size = nx * ny ;
  if ( pos < size )
    disp_image[pos] = 4 * d_find_min_index(&accumulated_costs[pos * disp_range], disp_range);
}

__device__ int d_find_min_index ( const int *v, const int disp_range )
{
    int min = INT_MAX;
    int minind = -1;
    for (int d=0; d < disp_range; d++) {
         if(v[d]<min) {
              min = v[d];
              minind = d;
         }
    }
    return minind;
}

/* functions code */

void determine_costs ( const int *left_image, const int *right_image, int *costs,
                       const int nx, const int ny, const int disp_range )
{
  std::fill(costs, costs+nx*ny*disp_range, 255u);

  for ( int j = 0; j < ny; j++ ) {
      for ( int d = 0; d < disp_range; d++ ) {
          for ( int i = d; i < nx; i++ ) {
              COSTS(i,j,d) = abs( LEFT_IMAGE(i,j) - RIGHT_IMAGE(i-d,j) );
        }
      }
  }
}

void iterate_direction_dirxpos ( const int dirx, const int *left_image,
                                 const int* costs, int *accumulated_costs,
                                 const int nx, const int ny, const int disp_range )
{
    const int WIDTH = nx;
    const int HEIGHT = ny;

      for ( int j = 0; j < HEIGHT; j++ ) {
          for ( int i = 0; i < WIDTH; i++ ) {
              if(i==0) {
                  for ( int d = 0; d < disp_range; d++ ) {
                      ACCUMULATED_COSTS(0,j,d) += COSTS(0,j,d);
                  }
              }
              else {
                  evaluate_path( &ACCUMULATED_COSTS(i-dirx,j,0),
                                 &COSTS(i,j,0),
                                 abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i-dirx,j)) ,
                                 &ACCUMULATED_COSTS(i,j,0), nx, ny, disp_range);
              }
          }
      }
}

void iterate_direction_dirypos ( const int diry, const int *left_image,
                                 const int* costs, int *accumulated_costs,
                                 const int nx, const int ny, const int disp_range )
{
    const int WIDTH = nx;
    const int HEIGHT = ny;

      for ( int i = 0; i < WIDTH; i++ ) {
          for ( int j = 0; j < HEIGHT; j++ ) {
              if(j==0) {
                  for ( int d = 0; d < disp_range; d++ ) {
                      ACCUMULATED_COSTS(i,0,d) += COSTS(i,0,d);
                  }
              }
              else {
                  evaluate_path( &ACCUMULATED_COSTS(i,j-diry,0),
                                 &COSTS(i,j,0),
                                 abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i,j-diry)),
                                 &ACCUMULATED_COSTS(i,j,0), nx, ny, disp_range );
              }
          }
      }
}

void iterate_direction_dirxneg ( const int dirx, const int *left_image,
                                 const int* costs, int *accumulated_costs,
                                 const int nx, const int ny, const int disp_range )
{
    const int WIDTH = nx;
    const int HEIGHT = ny;

      for ( int j = 0; j < HEIGHT; j++ ) {
          for ( int i = WIDTH-1; i >= 0; i-- ) {
              if(i==WIDTH-1) {
                  for ( int d = 0; d < disp_range; d++ ) {
                      ACCUMULATED_COSTS(WIDTH-1,j,d) += COSTS(WIDTH-1,j,d);
                  }
              }
              else {
                  evaluate_path( &ACCUMULATED_COSTS(i-dirx,j,0),
                                 &COSTS(i,j,0),
                                 abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i-dirx,j)),
                                 &ACCUMULATED_COSTS(i,j,0), nx, ny, disp_range );
              }
          }
      }
}

void iterate_direction_diryneg ( const int diry, const int *left_image,
                                 const int* costs, int *accumulated_costs,
                                 const int nx, const int ny, const int disp_range )
{
    const int WIDTH = nx;
    const int HEIGHT = ny;

      for ( int i = 0; i < WIDTH; i++ ) {
          for ( int j = HEIGHT-1; j >= 0; j-- ) {
              if(j==HEIGHT-1) {
                  for ( int d = 0; d < disp_range; d++ ) {
                      ACCUMULATED_COSTS(i,HEIGHT-1,d) += COSTS(i,HEIGHT-1,d);
                  }
              }
              else {
                  evaluate_path( &ACCUMULATED_COSTS(i,j-diry,0),
                           &COSTS(i,j,0),
                           abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i,j-diry)),
                           &ACCUMULATED_COSTS(i,j,0) , nx, ny, disp_range);
             }
         }
      }
}

/*
 *d_iterate_direction: computes iterate_direction_dirxpos() using the
 *the GPU
 */
void d_iterate_direction ( int* d_costs,
                           const int *d_left_image, int *d_accumulated_costs,
                           const int dirx, const int diry, //const int *left_image,
                           //const int* costs, int *accumulated_costs,
                           const int nx, const int ny, const int disp_range )
{
  dim3 block1d(1);
  dim3 grid1d(1);

  // Walk along the edges in a clockwise fashion
  if ( dirx > 0 ) {
    // LEFT MOST EDGE
    // Process every pixel along this edge
    //iterate_direction_dirxpos(dirx,left_image,costs,accumulated_costs, nx,
    // ny, disp_range);
    block1d.z = disp_range;
    grid1d.y = ny;
    d_iterate_direction_dirxpos <<< grid1d, block1d >>> ( dirx, d_left_image,
                                                          d_costs, d_accumulated_costs,
                                                          nx, ny, disp_range );
  }
  else if ( diry > 0 ) {
    // TOP MOST EDGE
    // Process every pixel along this edge only if dirx ==
    // 0. Otherwise skip the top left most pixel
    // iterate_direction_dirypos(diry,left_image,costs,accumulated_costs, nx,
    // ny, disp_range);
    block1d.z = disp_range;
    grid1d.y = nx;
    d_iterate_direction_dirypos <<< grid1d, block1d >>> ( diry, d_left_image,
                                                          d_costs, d_accumulated_costs,
                                                          nx, ny, disp_range );
  }
  else if ( dirx < 0 ) {
    // RIGHT MOST EDGE
    // Process every pixel along this edge only if diry ==
    // 0. Otherwise skip the top right most pixel
    //iterate_direction_dirxneg(dirx,left_image,costs,accumulated_costs, nx, ny, disp_range);

    block1d.z = disp_range;
    grid1d.y = ny;
    d_iterate_direction_dirxneg <<< grid1d, block1d >>> ( dirx, d_left_image,
                                                          d_costs, d_accumulated_costs,
                                                          nx, ny, disp_range );
  }
  else if ( diry < 0 ) {
    // BOTTOM MOST EDGE
    // Process every pixel along this edge only if dirx ==
    // 0. Otherwise skip the bottom left and bottom right pixel
    //iterate_direction_diryneg(diry,left_image,costs,accumulated_costs, nx, ny, disp_range);

    block1d.z = disp_range;
    grid1d.y = nx;
    d_iterate_direction_diryneg <<< grid1d, block1d >>> ( diry, d_left_image,
                                                          d_costs, d_accumulated_costs,
                                                          nx, ny, disp_range );
  }
}

void iterate_direction ( const int dirx, const int diry, const int *left_image,
                         const int* costs, int *accumulated_costs,
                         const int nx, const int ny, const int disp_range )
{
    // Walk along the edges in a clockwise fashion
    if ( dirx > 0 ) {
      // LEFT MOST EDGE
      // Process every pixel along this edge
      iterate_direction_dirxpos(dirx,left_image,costs,accumulated_costs, nx, ny, disp_range);
    }
    else if ( diry > 0 ) {
      // TOP MOST EDGE
      // Process every pixel along this edge only if dirx ==
      // 0. Otherwise skip the top left most pixel
      iterate_direction_dirypos(diry,left_image,costs,accumulated_costs, nx, ny, disp_range);
    }
    else if ( dirx < 0 ) {
      // RIGHT MOST EDGE
      // Process every pixel along this edge only if diry ==
      // 0. Otherwise skip the top right most pixel
      iterate_direction_dirxneg(dirx,left_image,costs,accumulated_costs, nx, ny, disp_range);
    }
    else if ( diry < 0 ) {
      // BOTTOM MOST EDGE
      // Process every pixel along this edge only if dirx ==
      // 0. Otherwise skip the bottom left and bottom right pixel
      iterate_direction_diryneg(diry,left_image,costs,accumulated_costs, nx, ny, disp_range);
    }
}

// ADD two cost images
void inplace_sum_views ( int * im1, const int * im2,
                         const int nx, const int ny, const int disp_range )
{
    int *im1_init = im1;
    while ( im1 != (im1_init + (nx*ny*disp_range)) ) {
      *im1 += *im2;
      im1++;
      im2++;
    }
}

int find_min_index ( const int *v, const int disp_range )
{
    int min = std::numeric_limits<int>::max();
    int minind = -1;
    for (int d=0; d < disp_range; d++) {
         if(v[d]<min) {
              min = v[d];
              minind = d;
         }
    }
    return minind;
}

void evaluate_path ( const int *prior, const int *local,
                     int path_intensity_gradient, int *curr_cost ,
                     const int nx, const int ny, const int disp_range )
{
  memcpy(curr_cost, local, sizeof(int)*disp_range);

  for ( int d = 0; d < disp_range; d++ ) {
    int e_smooth = std::numeric_limits<int>::max();
    for ( int d_p = 0; d_p < disp_range; d_p++ ) {
      if ( d_p - d == 0 ) {
        // No penality
        e_smooth = MMIN(e_smooth,prior[d_p]);
      } else if ( abs(d_p - d) == 1 ) {
        // Small penality
        e_smooth = MMIN(e_smooth,prior[d_p]+PENALTY1);
      } else {
        // Large penality
        e_smooth =
          MMIN(e_smooth,prior[d_p] +
                   MMAX(PENALTY1,
                            path_intensity_gradient ? PENALTY2/path_intensity_gradient : PENALTY2));
      }
    }
    curr_cost[d] += e_smooth;
  }

  int min = std::numeric_limits<int>::max();
  for ( int d = 0; d < disp_range; d++ ) {
        if (prior[d]<min) min=prior[d];
  }
  for ( int d = 0; d < disp_range; d++ ) {
        curr_cost[d]-=min;
  }
}

void create_disparity_view ( const int *accumulated_costs , int * disp_image,
                             const int nx, const int ny, const int disp_range )
{
  for ( int j = 0; j < ny; j++ ) {
    for ( int i = 0; i < nx; i++ ) {
      DISP_IMAGE(i,j) =
        4 * find_min_index( &ACCUMULATED_COSTS(i,j,0), disp_range );
    }
  }
}

/*
 * Links:
 * http://www.dlr.de/rmc/rm/en/desktopdefault.aspx/tabid-9389/16104_read-39811/
 * http://lunokhod.org/?p=1356
 */

// sgm code to run on the host
void sgmHost ( const int *h_leftIm, const int *h_rightIm,
               int *h_dispIm,
               const int w, const int h, const int disp_range )
{
  const int nx = w;
  const int ny = h;

  // Processing all costs. W*H*D. D= disp_range
  int *costs = (int *) calloc(nx*ny*disp_range,sizeof(int));
  if (costs == NULL) {
        fprintf(stderr, "sgm_cuda:"
                " Failed memory allocation(s).\n");
        exit(1);
  }

  determine_costs(h_leftIm, h_rightIm, costs, nx, ny, disp_range);

  int *accumulated_costs = (int *) calloc(nx*ny*disp_range,sizeof(int));
  int *dir_accumulated_costs = (int *) calloc(nx*ny*disp_range,sizeof(int));
  if (accumulated_costs == NULL || dir_accumulated_costs == NULL) {
        fprintf(stderr, "sgm_cuda:"
                " Failed memory allocation(s).\n");
        exit(1);
  }

  int dirx=0,diry=0;
  for(dirx=-1; dirx<2; dirx++) {
      if(dirx==0 && diry==0) continue;
      std::fill(dir_accumulated_costs, dir_accumulated_costs+nx*ny*disp_range, 0);
      iterate_direction( dirx,diry, h_leftIm, costs, dir_accumulated_costs, nx, ny, disp_range);
      inplace_sum_views( accumulated_costs, dir_accumulated_costs, nx, ny, disp_range);
  }
  dirx=0;
  for(diry=-1; diry<2; diry++) {
      if(dirx==0 && diry==0) continue;
      std::fill(dir_accumulated_costs, dir_accumulated_costs+nx*ny*disp_range, 0);
      iterate_direction( dirx,diry, h_leftIm, costs, dir_accumulated_costs, nx, ny, disp_range);
      inplace_sum_views( accumulated_costs, dir_accumulated_costs, nx, ny, disp_range);
  }

  free(costs);
  free(dir_accumulated_costs);

  create_disparity_view( accumulated_costs, h_dispIm, nx, ny, disp_range );

  free(accumulated_costs);
}

// sgm code to run on the GPU
void sgmDevice ( const int *h_leftIm, const int *h_rightIm,
                 int *h_dispIm,
                 const int w, const int h, const int disp_range )
{
  int nx = w;
  int ny = h;
  int image_size = nx * ny * sizeof(int); // size in bytes
  int costs_size = disp_range * image_size;
  //  int image_dim = nx * ny;
  int costs_dim = disp_range * nx * ny;

  /* launching the determine_costs() kernel */
  int *d_left_image;
  int *d_right_image;
  int *d_costs;

  //error = cudaMalloc ((void **) &d_left_image, image_size)
  cudaMalloc((void **) &d_left_image, image_size);
  cudaMalloc((void **) &d_right_image, image_size);
  cudaMalloc((void **) &d_costs, costs_size);
  cudaMemset(d_costs, 0, costs_size);

  cudaMemcpy(d_left_image, h_leftIm, image_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_right_image, h_rightIm, image_size, cudaMemcpyHostToDevice);

  int block_x = 32;
  int block_y = (disp_range >= 16) ? 16 : disp_range; // 32 * 16 = 512

  int z_blocks = (disp_range % block_y)
    ? ceil((float) disp_range / block_y) + 1
    : ceil((float) disp_range / block_y);

  int grid_x = ceil((float) nx / block_x) * z_blocks;
  int grid_y = ny;

  dim3 block(block_x, block_y);
  dim3 grid(grid_x, grid_y);

  d_determine_costs <<< grid, block >>> (d_left_image, d_right_image, d_costs,
                                        nx, ny, disp_range);

  int *accumulated_costs = (int *) calloc(nx*ny*disp_range,sizeof(int));
  //int *dir_accumulated_costs = (int *) calloc(nx*ny*disp_range,sizeof(int));
  if (accumulated_costs == NULL) {
    fprintf(stderr, "sgm_cuda:"
            " Failed memory allocation(s).\n");
    exit(1);
  }

  /* allocating space for dir_accumulated_costs and accumulated_costs on device */
  int *d_accumulated_costs;
  cudaMalloc((void **) &d_accumulated_costs, costs_size);
  cudaMemset( d_accumulated_costs, 0, costs_size);

  int *d_dir_accumulated_costs;
  cudaMalloc((void **) &d_dir_accumulated_costs, costs_size);

  // geometry for d_inplace_sum_views
  dim3 block1d(1);
  dim3 grid1d(1);
  if (costs_dim >= 512)
    {
      block1d.x = 512;
      grid1d.x = ceil((float) costs_dim/512);
    }
  else
    {
      block1d.x = costs_dim;
      grid1d.x = 1;
    }

  int dirx=0,diry=0;
  for(dirx=-1; dirx<2; dirx++) {
    if(dirx==0 && diry==0) continue;
    cudaMemset( d_dir_accumulated_costs, 0, costs_size);
    d_iterate_direction ( d_costs,
                          d_left_image, d_dir_accumulated_costs,
                          dirx, diry,
                          nx, ny, disp_range );

    d_inplace_sum_views <<< grid1d, block1d >>> ( d_accumulated_costs, d_dir_accumulated_costs,
                                                  nx, ny, disp_range);
  }

  dirx=0;
  for(diry=-1; diry<2; diry++) {
    if(dirx==0 && diry==0) continue;
    cudaMemset( d_dir_accumulated_costs, 0, costs_size);
    d_iterate_direction ( d_costs,
                          d_left_image, d_dir_accumulated_costs,
                          dirx, diry,
                          nx, ny, disp_range );
    d_inplace_sum_views <<< grid1d, block1d >>> ( d_accumulated_costs, d_dir_accumulated_costs,
                                              nx, ny, disp_range);
  }

  cudaMemcpy(accumulated_costs, d_accumulated_costs, costs_size, cudaMemcpyDeviceToHost);

  //free(costs);

  // device memory mgmt

  cudaFree(d_dir_accumulated_costs);
  cudaFree(d_left_image);
  cudaFree(d_right_image);
  cudaFree(d_costs);
  cudaFree(d_accumulated_costs);

  create_disparity_view( accumulated_costs, h_dispIm, nx, ny, disp_range );

  free(accumulated_costs);
}

// print command line format
void usage ( char *command )
{
    printf("Usage: %s [-h] [-d device] [-l leftimage] [-r rightimage] [-o dev_dispimage] [-t host_dispimage] [-p disprange] \n",command);
}

// main
int
main ( int argc, char** argv )
{

    // default command line options
    int deviceId = 0;
    int disp_range = 32;
    char *leftIn      =(char *)"lbull.pgm",
         *rightIn     =(char *)"rbull.pgm",
         *fileOut     =(char *)"d_dbull.pgm",
         *referenceOut=(char *)"h_dbull.pgm";

    // parse command line arguments
    int opt;
    while( (opt = getopt(argc,argv,"d:l:o:r:t:p:h")) !=-1)
    {
        switch(opt)
        {

            case 'd':  // device
                if(sscanf(optarg,"%d",&deviceId)!=1)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;

            case 'l': // left image filename
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }

                leftIn = strdup(optarg);
                break;
            case 'r': // right image filename
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }

                rightIn = strdup(optarg);
                break;
            case 'o': // output image (from device) filename
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                fileOut = strdup(optarg);
                break;
            case 't': // output image (from host) filename
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                referenceOut = strdup(optarg);
                break;
            case 'p': // disp_range
                if(sscanf(optarg,"%d",&disp_range)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;
            case 'h': // help
                usage(argv[0]);
                exit(0);
                break;

        }

    }
    if(optind < argc) {
        fprintf(stderr,"Error in arguments\n");
        usage(argv[0]);
        exit(1);
    }

    // select cuda device
    cutilSafeCall( cudaSetDevice( deviceId ) );

    // create events to measure host sgm time and device sgm time
    cudaEvent_t startH, stopH, startD, stopD;
    cudaEventCreate(&startH);
    cudaEventCreate(&stopH);
    cudaEventCreate(&startD);
    cudaEventCreate(&stopD);

    // allocate host memory
    int* h_ldata=NULL;
    int* h_rdata=NULL;
    unsigned int h,w;

    //load left pgm
    if (cutLoadPGMi(leftIn, (unsigned int **)&h_ldata, &w, &h) != CUTTrue) {
        printf("Failed to load image file: %s\n", leftIn);
        exit(1);
    }
    //load right pgm
    if (cutLoadPGMi(rightIn, (unsigned int **)&h_rdata, &w, &h) != CUTTrue) {
        printf("Failed to load image file: %s\n", rightIn);
        exit(1);
    }

    // allocate mem for the result on host side
    int* h_odata = (int*) malloc( h*w*sizeof(int));
    int* reference = (int*) malloc( h*w*sizeof(int));

    // sgm at host
    cudaEventRecord( startH, 0 );
    sgmHost(h_ldata, h_rdata, reference, w, h, disp_range);
    cudaEventRecord( stopH, 0 );
    cudaEventSynchronize( stopH );

    // sgm at GPU
    cudaEventRecord( startD, 0 );
    sgmDevice(h_ldata, h_rdata, h_odata, w, h, disp_range);
    cudaEventRecord( stopD, 0 );
    cudaEventSynchronize( stopD );

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    float timeH, timeD;
    cudaEventElapsedTime( &timeH, startH, stopH );
    printf( "Host processing time: %f (ms)\n", timeH);
    cudaEventElapsedTime( &timeD, startD, stopD );
    printf( "Device processing time: %f (ms)\n", timeD);
    printf( "SpeedUp: %f (ms)\n", timeH /timeD);

    // save output images
    // save output images
    if (cutSavePGMi(referenceOut, (unsigned int *)reference, w, h) != CUTTrue) {
        printf("Failed to save image file: %s\n", referenceOut);
        exit(1);
    }
    if (cutSavePGMi(fileOut,(unsigned int *) h_odata, w, h) != CUTTrue) {
        printf("Failed to save image file: %s\n", fileOut);
        exit(1);
    }

    // cleanup memory
    cutFree( h_ldata);
    cutFree( h_rdata);
    free( h_odata);
    free( reference);

    cutilDeviceReset();
}
