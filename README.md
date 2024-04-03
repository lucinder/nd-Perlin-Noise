# Mass Parallelization of n-Dimensional Perlin-Like Noise with CUDA
Project by [lucinder](https://github.com/lucinder) and [enbaik](https://github.com/enbaik)

## I - Introduction

Perlin noise is a procedural noisy texture generation algorithm designed by Ken Perlin,
originally developed as the Pixel Stream Editor in 1985 [[1]](#cite1)[[2]](#cite2). Perlin later developed Simplex
Noise, which improved this to utilize a simpler space-filling grid in 2001 [3](#cite3). Traditionally, this
algorithm is applied to terrain and particle generation programs, particularly in game
development and landscape ecology [[4]](#cite4)[[5]](#cite5)[[6]](#cite6), though it has extensive applications in other fields
such as microbiology and material sciences [[7]](#cite7)[[8]](#cite8)[[9]](#cite9)[[10]](#cite10). Figures 1 and 2 show examples of the
Perlin noise algorithm applied to terrain generation, with Figure 2 using a variant known as value
noise

*Figure 1: Procedural Terrain Generation Using Perlin Noise*

*Figure 2: Procedural Terrain Generation Using Value Noise [[11]](#cite11)*

With the rising popularity of machine learning and deep learning algorithms for various
applications, noise generation algorithms are essential in adversarial machine learning
[[12]](#cite12)[[13]](#cite13)[[14]](#cite14), with potential applications in defending against model poisoning attacks [[15]](#cite15). In
areas of machine learning, Perlin noise has already seen some applications, particularly in data
augmentation [[16]](#cite16)[[17]](#cite17)[[18]](#cite18). However, an overlooked factor for scaling this algorithm to modern
machine learning algorithms is the dimensionality of the noise in relation to the data, which in
many ML systems may exist in much higher dimensionality than 2D or 3D.

Few studies have investigated the parallelization of Perlin noise generation in CUDA
[[19]](#cite19)[[20]](#cite20), none of which have scaled the Perlin noise algorithm to n dimensionality in their
implementations. As an unexplored avenue of research, creating a parallelized nD
implementation of Perlin noise could lay a foundation for improved noise generation applications
in data augmentation and adversarial machine learning, where data dimensionality may be much
higher than the current three-dimensional limit.

## II - Perlin Noise Algorithm

The Perlin Noise algorithm can be divided into these major sections for each point in the
n-dimensional matrix:
1. Generate distance vectors from each point to its grid space’s edges.
2. Generate the appropriate pseudo-random gradient vectors for each edge.
3. Calculate the dot product of each distance vector with each gradient vector.
4. Repeatedly perform interpolation in each dimension between dot products until only one
value remains.

In subsections here, we explain the theoretical implementation, pseudocode, and
n-dimensional scaling behind each step, including the additional fade function applied during
step 4. In section III, we explain in depth the constraints of a CUDA implementation of this
algorithm.

### II.A. Distance Vector Generation

For a given 2-dimensional m x m grid, a given point located at (x, y) has four edges: (x0,
y0), (x1, y0), (x0, y1), and (x1, y1), where x0 = |x|, y0 = |y|, x1 = (x0+1)%m, and y1 = (y0+1)%m.
Generalizing this to n dimensions, we have (x0, y0, z0, …, pn,0), (x1, y0, z0, …, pn,0), … (x1, y1, z1,
…, pn,1). Each n-dimensional point has 2n edges and, therefore, 2n distance vectors.
We save the effort of calculating and storing all possible combinations of edges by
treating our point’s coordinates in the grid space as a fraction of the grid limits- that is, for a grid
of size mn, we treat our coordinate set as (x/m, y/m, z/m, ... in/m). When treated as such, to
calculate distances, our edge vectors are (0, 0, 0, … 0), (0, 0, 0, … 1), … (1, 1, 1, … 1); the
digits of the set of all binary numbers of length n.

Our pseudocode for n-dimensional distance vector generation is as follows:
```
distances = empty list
for i in [0, 2n): // loop through edges
  for j in [0, n): // loop through dimensions
    edge = ((i & (1 << (j - 1))) >> (j - 1)); // get the jth digit of binary i
    distances[i*n+j] = point[j] - edge // take the distance
```

### II.B. Gradient Vector Generation

Expanding gradient vector generation to n dimensions proves a more difficult task, as the
original and improved Perlin Noise algorithms use a hard-coded list of gradients, the selection of
which is determined by a hash function. Generating these lists of gradients in higher
dimensionality requires an exponentially increasing space complexity, just as with the distance
vectors.

The gradient vectors generally consist of pseudo-random components determined by a
hash function. We encountered difficulty understanding and implementing the gradient function
in a fashion consistent with other writings, as many implementations offer different gradient
generation methods. However, our final implementation uses the hash function’s
pseudo-randomness to map block and thread offsets to random seeding. It uses the hashed seed
to randomly generate a set of gradient values from -1 to 1. Once generated, we divide these
values by the Euclidean distance of the resulting vector to normalize the vector to a length of 1,
mapping it to a n-dimensional unit sphere.

Our pseudocode for n-dimensional gradient generation is as follows:
```
gradients = empty list
srand with hash[(blockDim.x * blockIdx.x + threadIdx.x)%len(hash)]
for i in [0,2n):
  euclidean_distance = 0
  for i in [0, n): // loop through dimensions
    gradients[i*n+j] = rand in (-1, 1)
    euclidean_distance += gradients[i*n+j]2
  euclidean_distance = sqrt(euclidean_distance)
  for i in [0, n):
    gradients[i*n+j] = gradients[i*n+j] / euclidean_distance
```

### II.C. Dot Product

In this step, we take the dot products of the gradient and distance vectors of each edge to
acquire an “influence” vector for the whole matrix point, showing the “influences” of each edge
on the point with pseudo-randomization from the gradients. This is a fast step, taking O(n)
operations for each point. For any two n-dimensional vectors and , we define the dot product
𝑎
→
𝑏
→
as:
$$\left(\vec{a}⋅\vec{b}\right) \eq \left(a_1 b_1+a_2 b_2+a_3 b_3+…+a_n b_n\right)$$

Or,

$$\left(\vec{a}⋅\vec{b}\right) \eq \left(\sum_{i=1}^n a_i b_i\right)$$

Our pseudocode for the dot product of the distance and gradient vectors is as follows:
```
products = empty list
for i in [0, 2n):
  dot = 0
  for j in [0, n):
    dot += distances[i*n+j] * gradients[i*n+j]
  products[i] = dot
```

### II.D. Interpolation

Two different types of interpolation are commonly used to generate Perlin noise: linear
interpolation and cosine interpolation. Cosine interpolation is typically preferred as it provides a
smoother fade of different values across the grid. However, we will use linear interpolation for
our implementation, as we are primarily interested in performance. Our implementation can be
easily modified to use cosine interpolation instead of linear interpolation.

Given that we do not have a set dimensionality, our literal number of iterations for
interpolation is unknown but can be modeled as $\sum_{i=0}^n-1 2^i$. Thus, for n = 1, we have one interpolation;
for n = 2, we have three interpolations; for n = 3, we have 7, etc. For a given vector, we can start
with interpolations of values at a step of 1, saving the results into the former, and increase our
step by a factor of 2 each time until it exceeds the length of the vector. At this point, we return
the resulting value at index 0.

Our pseudocode for the interpolation is as follows:

```
step = 1, dimension = 0
while step < 2n:
  for i in [0, 2n] incrementing by 2*step:
    fraction = point[dimension]
    products[i] = products[i] + fraction * (products[i+step] - products[i])
  step *= 2
  dimension++ // move to the next dimension
```

### II.E. Fade Function

The fade function defined by Perlin, $ψ(t) = 6t^5 − 15t^4 + 10t^3$, allows the gradient to “fade”
as the displacement grows further from the edges, preventing disproportionate influences of the
edges [[21]](#cite21). This function was originally defined as the Hermite blending function $ψ(t) = 3t2 - 2t3$
but later revised to the fifth-degree polynomial to ensure a continuous second derivative [[3]](#cite3). We
can modify our interpolation function to incorporate the fade function implicitly as such:

```
step = 1, dimension = 0
while step < 2n:
  for i in [0, 2n] incrementing by 2*step:
    f = point[dimension] // our fractional displacement
    f = f*f*f*(f*(f*6-15)+10) // apply fade function
    prods[i] = prods[i] + f * (prods[i+step] - prods[i])
  step *= 2
  dimension++ // move to the next dimension
```

### II.F. Full Pseudocode Without Optimizations

Combining all parts of the pseudocode thus far, our pseudocode for calculating the Perlin
noise value for a given point is as follows:
```
perlin(point):
  // generate gradients and distances
  distances = empty list
  gradients = empty list
  srand with hash[(blockDim.x * blockIdx.x + threadIdx.x)%256]
  for i in [0, 2n): // loop through edges
    euclidean distance = 0
    for j in [0, n): // loop through dimensions
      index = i*n+j
      gradients[index] = rand in (-1, 1)
      euclidean distance += gradients[index]2
      edge = (i & (1 << (j - 1))) >> (j - 1)); // get the jth digit of binary i
      distances[index] = edge - point[j] // take the distance
    euclidean distance = sqrt(euclidean distance)
    for j in [0, n):
      gradients[index] = gradient[index] / euclidean distance
  // calculate dot product
  prods = empty list
  for i in [0, 2n):
    dot = 0
    for j in [0, n):
      dot += edges[i*n+j] * gradients[i*n+j] // take dot product
    products[i] = dot
  step = 1, dimension = 0
  while step < 2n:
    for i in [0, 2n] incrementing by 2*step:
      f = point[dimension] // our fractional displacement
      f = f*f*f*(f*(f*6-15)+10) // apply fade function
      products[i] = products[i] + f * (products[i+step] - products[i])
    step *= 2
    dimension++ // move to the next dimension
  return products[0] // return the collapsed dot product
```

### II.G. Trivial Optimizations and Improved Pseudocode

As is, this is a very inefficient implementation of the noise function in terms of memory,
with each point requiring two n*2n length vectors (gradients and edges) and two n-length vectors
(point and dot products). We can eliminate much of the memory overhead by reusing registers
and vector space for multiple operations. For example, the gradients vector can be used to hold
the dot products while a set-length vector loads the current gradient per each edge. In the linear
interpolation step, we can collapse each interpolation along the steps in the gradient vector,
holding the dot products until the final interpolation rests at gradients\[0\]. Additionally, we can
forgo creating a distances vector as distances are only used once per edge, and instead, we can
use their formula implicitly while calculating dot products. The following shows an improved
version of the pseudocode using trivial memory optimizations.
```
perlin(point):
  gradients = empty list
  srand with hash[(blockDim.x * blockIdx.x + threadIdx.x)%256]
  // calculate gradients
  for i in [0, 2n): // loop through edges
    current_gradient = empty list
    euclidean_distance = 0
    for j in [0, n):
      current_gradent[j] = rand in (-1, 1)
    // calculate distances and dot products
    edge = ((i & (1 << -1)) >> - 1);
    gradients[i] = (current_gradient[0] / euclidean_distance) * (point[0]-edge); // unroll first addition operation into initialization
    for j in [1, n):
      edge = ((i & (1 << (j - 1))) >> (j - 1)); // get the jth digit of binary i
      gradients[i] += (point[j]- edge) * (current_gradient[j]/euclidean_distance)
    // linear interpolation
  step = 1, dimension = 0
  while step < 2n:
    for i in [0, 2n) incrementing by 2*step:
      f = point[dimension] // our fractional displacement
      f = f*f*f*(f*(f*6-15)+10) // apply fade function
      gradients[i] = gradients[i] + f * (gradients[i+step] - gradients[i])
    step *= 2
    dimension++ // move to the next dimension
  return gradients[0]
```

## III - Hypotheses and System Limitations

### III.A. System Specifications

For measuring runtimes, we ran our code with the following system specifications:
- CPU: AMD Ryzen 7 4800H
- GPU: NVIDIA GeForce RTX 3060 Laptop
- Compute Capability [[22]](#cite22): 8.6
- Architecture [[23]](#cite23): Ampere
- 30 Streaming Multiprocessors
- Software: Visual Studio 2022 v. 17.9.3, CUDA v. 12.3
- CPU Memory: 16 GB RAM, 475 GB HDD
- GPU Memory: 6 GB, 192-bit Memory Interface Width

For running the profiler, we ran our code on a separate system with the following
specifications:
- CPU: Intel® Core™ i9-10920X -12 Core -3.5GHz Processor
- GPU: NVIDIA GeForce RTX 3070
- Compute Capability [[22]](#cite22): 8.6
- Architecture: Ampere
- 46 Streaming Multiprocessors
- Software: Nsight Compute v. 2024.1, CUDA v. 12.3
- CPU Memory: 32 GB RAM, 1 TB HDD
- GPU Memory: 8 GB, 256-bit Memory Interface Width
  
With Compute Capability 8.6 on both systems, we also observe the following technical
specifications [[24]](#cite24):
- Maximum 128 resident grids per device.
- Maximum 16 resident blocks, 1536 resident threads, and 48 resident warps per SM.
- Maximum 64K registers per thread block, 255 registers per thread.
- For 1-D grids/blocks:
- 231-1 maximum blocks per grid.
- 1024 maximum threads per block.
- Warp size 32.
  
With this in mind, our system should be able to host over $2.19\times10^12$ total threads on a
1-dimensional grid and execute 1536 threads concurrently per SM. For a given matrix size $1024
< mn < 2.19*10^12$, we need $m^n/1,572,864$ warps to fully execute our noise generation on the
device.

### III.B. Memory & Register Use

One of the primary limitations on the feasibility of parallelizing Perlin noise is the space
complexity needed to hold all relevant data in thread registers. While the baseline matrix of size
*m* and dimensionality *n* will only generate $O(m^n)$ floating-point noise values, the intermediate
spaces needed to hold the points on the matrix ($O(n\timesm^n)$ floating-point values) and the influence
vectors ($O(2^n m^n)$ floating-point values) result in significant memory usage. Table 1 shows the
memory space needed for different grid sizes and dimensionality, where row R/T is the registers
per thread necessary for the coordinates, influences, and intermediate gradient vector ($2 n+2^n$).
