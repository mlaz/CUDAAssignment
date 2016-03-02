
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

#define COSTS(i,j,d)               costs[(i)*disp_range+(j)*nx*disp_range+(d)]
#define ACCUMULATED_COSTS(i,j,d)   accumulated_costs[(i)*disp_range+(j)*nx*disp_range+(d)]
#define LEFT_IMAGE(i,j)            left_image[(i)+(j)*nx]
#define RIGHT_IMAGE(i,j)           right_image[(i)+(j)*nx]
#define DISP_IMAGE(i,j)            disp_image[(i)+(j)*nx]
#define SHR_ACCUMULATED_COSTS(i,d) shr_accumulated_costs[(i)+(d)*disp_range]

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

__device__ int d_find_min_index ( const int *v, const int disp_range);

__device__ void d_evaluate_path ( int *prior, int *local,
                                int path_intensity_gradient, int *curr_cost,
                                  int nx, int ny, int disp_range, int *tmp);

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

__global__ void d_iterate_direction ( int* costs,
                                      const int *left_image, int *accumulated_costs,
                                      const int dirx, const int diry,
                                      const int nx, const int ny, const int disp_range )
{
  extern __shared__ int shr[];

  // Walk along the edges in a clockwise fashion
  if ( dirx != 0 ) {
    int* sh_current_l = shr;
    int* sh_prior_l = sh_current_l + disp_range;
    int* sh_tmp_l = sh_prior_l + disp_range;

    int* sh_current_r = sh_tmp_l + disp_range;
    int* sh_prior_r = sh_current_r + disp_range;
    int* sh_tmp_r = sh_prior_r + disp_range;

    int* aux;

    int x_l = nx-1;
    int x_r = 0;
    int y = blockIdx.y;
    int d = threadIdx.z;

    if ( (y < ny) && (d < disp_range) )
      {
        if (blockIdx.x == 0)
          {
            // RIGHT DIRECTION
            // Proce    ss every pixel along this edge
            /* copying the first vector to shared memory */
            __syncthreads();
            sh_current_r[d] = COSTS(0,y,d);
            ACCUMULATED_COSTS(0,y,d) += COSTS(0,y,d);
          }
        else
          {
            // LEFT DIRECTION
            // Process every pixel along this edge only if diry ==
            // 0. Ot    herwise skip the top right most pixel
            /* copying the first vector to shared memory */
            __syncthreads();
            sh_current_l[d] = COSTS(nx-1,y,d);
            ACCUMULATED_COSTS(nx-1,y,d) += COSTS(nx-1,y,d);
          }

        for ( x_r = 1, x_l = nx-2; x_r < nx && x_l > 0; x_r++, x_l-- )
          {
            __syncthreads();
            if (blockIdx.x == 0)
              {
                // RIGHT DIRECTION
                /*  swap vectors and read next vector from global memory*/
                aux = sh_current_r;
                sh_current_r = sh_prior_r;
                sh_prior_r = aux;
                sh_current_r[d] = ACCUMULATED_COSTS(x_r,y,d);
                __syncthreads();

                d_evaluate_path( sh_prior_r,
                                 &COSTS(x_r,y,0),
                                 abs(LEFT_IMAGE(x_r,y)-LEFT_IMAGE(x_r-dirx,y)) ,
                                 sh_current_r, nx, ny, disp_range, sh_tmp_r);
                __syncthreads();

                if (x_r != x_l)
                  ACCUMULATED_COSTS(x_r,y,d) += sh_current_r[d];
                else
                  ACCUMULATED_COSTS(x_r,y,d) += sh_current_r[d] + sh_current_l[d];
              }
            else
              {
                // LEFT DIRECTION
                /* swap vectors and read next vector from global memory*/
                aux = sh_current_l;
                sh_current_l = sh_prior_l;
                sh_prior_l = aux;
                sh_current_l[d] = ACCUMULATED_COSTS(x_l,y,d);
                __syncthreads();

                d_evaluate_path( sh_prior_l,
                                 &COSTS(x_l,y,0),
                                 abs(LEFT_IMAGE(x_l,y)-LEFT_IMAGE(x_l+dirx,y)) ,
                                 sh_current_l, nx, ny, disp_range, sh_tmp_l);
                __syncthreads();
                if(x_r != x_l)
                  ACCUMULATED_COSTS(x_l,y,d) += sh_current_l[d];
              }
          }

      }
  }
  else if ( diry != 0 ) {
    int* sh_current_u = shr;
    int* sh_prior_u = sh_current_u + disp_range;
    int* sh_tmp_u = sh_prior_u + disp_range;

    int* sh_current_d = sh_tmp_u + disp_range;
    int* sh_prior_d = sh_current_d + disp_range;
    int* sh_tmp_d = sh_prior_d + disp_range;

    int* aux;

    int x = blockIdx.y;
    int y_u = ny-1;
    int y_d = 0;
    int d = threadIdx.z;

    if ( (x < nx) && (d < disp_range) )
      {
        if (blockIdx.x == 0)
          {
            // DOWNWARDS DIRECTION
            // Process every pixel along this edge
            /* copying the first vector to shared memory */
            __syncthreads();
            sh_current_d[d] = COSTS(x,0,d);
            ACCUMULATED_COSTS(x,0,d) += COSTS(x,0,d);

          }
        else
          {
            // UPWARDS DIRECTION
            // Process every pixel along this edge only if diry ==
            // 0. Ot    herwise skip the top right most pixel
            /* copying the first vector to shared memory */

            __syncthreads();
            sh_current_u[d] = COSTS(x,ny-1,d);
            ACCUMULATED_COSTS(x,ny-1,d) += COSTS(x,ny-1,d);
          }

        for ( y_d = 1, y_u = ny-2; y_u > 0 && y_d < ny; y_d++, y_u-- )
          {
            __syncthreads();
            if (blockIdx.x == 0)
              {
                // DOWNWARDS DIRECTION
                /*  swap vectors and read next vector from global memory*/
                aux = sh_current_d;
                sh_current_d = sh_prior_d;
                sh_prior_d = aux;
                sh_current_d[d] = ACCUMULATED_COSTS(x,y_d,d);


                d_evaluate_path( sh_prior_d,
                                 &COSTS(x,y_d,0),
                                 abs(LEFT_IMAGE(x,y_d)-LEFT_IMAGE(x,y_d-diry)) ,
                                 sh_current_d, nx, ny, disp_range, sh_tmp_d);
                __syncthreads();

                if (y_u != y_d)
                  ACCUMULATED_COSTS(x,y_d,d) += sh_current_d[d];
                else
                  ACCUMULATED_COSTS(x,y_d,d) += sh_current_d[d] + sh_current_u[d];
              }
            else
              {
                // UPWARDS DIRECTION
                /* swap vectors and read next vector from global memory*/
                aux = sh_current_u;
                sh_current_u = sh_prior_u;
                sh_prior_u = aux;
                sh_current_u[d] = ACCUMULATED_COSTS(x,y_u,d);
                __syncthreads();

                d_evaluate_path( sh_prior_u,
                                 &COSTS(x,y_u,0),
                                 abs(LEFT_IMAGE(x,y_u)-LEFT_IMAGE(x,y_u+diry)) ,
                                 sh_current_u, nx, ny, disp_range, sh_tmp_u);
                __syncthreads();
                if (y_u != y_d)
                  ACCUMULATED_COSTS(x,y_u,d) += sh_current_u[d];
              }
          }

      }
  }

  // else if ( diry > 0 ) {
  //   int* sh_current = shr;
  //   int* sh_prior = sh_current + disp_range;
  //   int* sh_tmp = sh_prior + disp_range;

  //   int* aux;

  //   // TOP MOST EDGE
  //   // Process every pixel along this edge only if dirx ==
  //   // 0. Otherwise skip the top left most pixel
  //   int x = blockIdx.y;
  //   int y = 0;
  //   int d = threadIdx.z;

  //   if ( (x < nx) && (d < disp_range) )
  //     {
  //       /* copying the first vector to shared memory */
  //       ACCUMULATED_COSTS(x,0,d) += COSTS(x,0,d);
  //       __syncthreads();
  //       sh_current[d] = ACCUMULATED_COSTS(x,0,d);

  //       for (y = 1; y < ny; y++)
  //         {
  //           /* swap vectors and read next vector from global memory*/
  //           aux = sh_current;
  //           sh_current = sh_prior;
  //           sh_prior = aux;
  //           sh_current[d] = ACCUMULATED_COSTS(x,y,d);
  //           __syncthreads();

  //           d_evaluate_path( sh_prior,
  //                            &COSTS(x,y,0),
  //                            abs(LEFT_IMAGE(x,y)-LEFT_IMAGE(x,y-diry)) ,
  //                            sh_current, nx, ny, disp_range, sh_tmp);
  //           ACCUMULATED_COSTS(x,y,d) += sh_current[d];
  //           __syncthreads();
  //         }
  //     }
  // }

  // else if ( diry < 0 ) {
  //   // BOTTOM MOST EDGE
  //   // Process every pixel along this edge only if dirx ==
  //   // 0. Otherwise skip the bottom left and bottom right pixel
  //   int* sh_current = shr;
  //   int* sh_prior = sh_current + disp_range;
  //   int* sh_tmp = sh_prior + disp_range;

  //   int* aux;

  //   int x = blockIdx.y;
  //   int y = ny-1;
  //   int d = threadIdx.z;

  //   if ( (x < nx) && (d < disp_range) )
  //     {
  //       /* copying the first vector to shared memory */
  //       ACCUMULATED_COSTS(x,ny-1,d) += COSTS(x,ny-1,d);
  //       __syncthreads();
  //       sh_current[d] = ACCUMULATED_COSTS(x,ny-1,d);


  //       for (y = ny-2; y >= 0; y--)
  //         {
  //           /* swap vectors and read next vector from global memory*/
  //           aux = sh_current;
  //           sh_current = sh_prior;
  //           sh_prior = aux;
  //           sh_current[d] = ACCUMULATED_COSTS(x,y,d);
  //           __syncthreads();

  //           d_evaluate_path( sh_prior,
  //                            &COSTS(x,y,0),
  //                            abs(LEFT_IMAGE(x,y)-LEFT_IMAGE(x,y-diry)) ,
  //                            sh_current, nx, ny, disp_range, sh_tmp);
  //           ACCUMULATED_COSTS(x,y,d) += sh_current[d];
  //           __syncthreads();
  //         }
  //     }
  // }

}
__device__ void d_evaluate_path ( int *prior, int *local,
                                  int path_intensity_gradient, int *curr_cost,
                                  int nx, int ny, int disp_range, int *tmp )
{
  int d = threadIdx.z;
  curr_cost[d] = local[d];
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

  tmp[d] = prior[d];
  __syncthreads();
  int dim = disp_range;
  do
    {
      dim = ceil((float)dim /2);
      if (d < dim)
        {
          if (prior[d + dim] < tmp[d])
            tmp[d] = prior[d + dim];
        }
      __syncthreads();
    } while (dim > 1);

  curr_cost[d] -= tmp[0];
  __syncthreads();
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
  int pos = ((blockIdx.x * blockDim.x) + threadIdx.x); //position in image
  int size = nx * ny;
  int d = threadIdx.z;
  int idx;

  if ( pos < size && d < disp_range )
    {
      idx = d_find_min_index(&accumulated_costs[pos * (disp_range)], disp_range);
      __syncthreads();
      if (d == 0)
        disp_image[pos] = 4 * idx;
    }
}

__device__ int d_find_min_index ( const int *v, const int disp_range )
{
  extern __shared__ int shr[];
  int minind = -1;
  int* idx = shr + (threadIdx.x * disp_range); //indexes array
  int d = threadIdx.z;
  idx[d] = d;
  int dim = ceil((float)disp_range/2);
  idx[d + dim] = d + dim;
  __syncthreads();

  dim =  disp_range;
  do
    {
      dim = ceil((float)dim/2);
      if (d < dim)
        {
          if (v[ idx[d + dim] ] < v[idx[d]] )
            idx[d] = idx[d + dim];
          if (v[ idx[d + dim] ] == v[idx[d]] && idx[d + dim] < idx[d])
            idx[d] = idx[d + dim];
        }
      __syncthreads();

    } while (dim > 1);

  minind = (v[idx[0]] < INT_MAX) ? idx[0] : minind;

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
  int image_dim = nx * ny;
  int costs_dim = disp_range * nx * ny;

  // cudaError error;

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

  /* allocating space for dir_accumulated_costs and accumulated_costs on device */
  int *d_accumulated_costs;
  cudaMalloc((void **) &d_accumulated_costs, costs_size);
  cudaMemset( d_accumulated_costs, 0, costs_size);

  // int *d_dir_accumulated_costs;
  // cudaMalloc((void **) &d_dir_accumulated_costs, costs_size);

  /* geometry for d_inplace_sum_views kernel */
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

  /* geometry for d_iterate_direction kernel */
  dim3 block2d_dir(2);
  dim3 block1d_dir(1);
  dim3 grid1d_dir(1);
  block1d_dir.z = disp_range;

  block1d_dir.x = 2;
  block2d_dir.z = disp_range;
  int dirx=0,diry=0;
  // for (dirx=-1; dirx<2; dirx++) {
  //   if (dirx==0 && diry==0)
  //     continue;

  dirx=1;
  diry=0;
  grid1d_dir.y = ny;
  cudaMemset( d_accumulated_costs, 0, costs_size);
  d_iterate_direction <<< grid1d_dir, block2d_dir, 6*disp_range*sizeof(int) >>>
    ( d_costs,
      d_left_image,
      d_accumulated_costs,
      dirx, diry,
      nx, ny, disp_range );

    // d_inplace_sum_views <<< grid1d, block1d >>> ( d_accumulated_costs,
    //                                               d_dir_accumulated_costs,
    //                                               nx, ny, disp_range);

   // }

  dirx=0;
  diry=1;
  // for (diry=-1; diry<2; diry++) {
  //   if (dirx==0 && diry==0)
  //     continue;
  //   grid1d_dir.y = nx;

    d_iterate_direction <<< grid1d_dir, block2d_dir, 6*disp_range*sizeof(int) >>>
      ( d_costs,
        d_left_image,
        d_accumulated_costs,
        dirx, diry,
        nx, ny, disp_range );

    // d_inplace_sum_views <<< grid1d, block1d >>> ( d_accumulated_costs,
    //                                               d_dir_accumulated_costs,
    //                                               nx, ny, disp_range);
  
  // device memory mgmt
  // cudaFree(d_dir_accumulated_costs);
  cudaFree(d_left_image);
  cudaFree(d_right_image);
  cudaFree(d_costs);

  /* geometry for create_disparity_view*/
  dim3 block2d(2);
  dim3 grid1d_cdv(1);
  block2d.z = ceil((float)disp_range/ 2);
  block2d.x = 7; //with a 64pixel disp_range the ammount of shared memory
                 //needed is 64 * 8 * 32 = 16k bytes
  grid1d_cdv.x = ceil((float) image_dim/block2d.x);

  int *d_disp_im;
  cudaMalloc((void **) &d_disp_im, image_size);

  d_create_disparity_view <<< grid1d_cdv, block2d, 7*disp_range*sizeof(int) >>>
    ( d_accumulated_costs, d_disp_im, nx, ny, disp_range );

  cudaMemcpy(h_dispIm, d_disp_im, image_size, cudaMemcpyDeviceToHost);

  cudaFree(d_disp_im);
  cudaFree(d_accumulated_costs);
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
    printf( "SpeedUp: %f (ms)\n", timeH/timeD);

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
